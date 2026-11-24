"""Phần dùng chung của PPO độc lập: đường dẫn, cấu hình, nạp bài toán, đặc trưng tĩnh, sinh dữ liệu huấn luyện.

Chỉ phụ thuộc vào định nghĩa bài toán chung của dự án:
    moo/problem.py  (Problem, State, sortie_metrics)  - luật chơi / mô phỏng
    moo/utils.py    (to_solution, verify_solution, ...) - kiểm tra lời giải độc lập
    data/data.py    (build_core, build_instance)      - bộ sinh instance chính thức
Không dùng bất cứ thứ gì trong rl/core.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RL_DIR = ROOT / "rl"
PPO_DIR = RL_DIR / "ppo"
MOO_DIR = ROOT / "moo"
DATA_ROOT = ROOT / "data" / "data"
CONFIG_PATH = RL_DIR / "config.yaml"
if str(MOO_DIR) not in sys.path:
    sys.path.insert(0, str(MOO_DIR))

from problem import Problem, State, sortie_metrics  # noqa: E402,F401

EPS = 1e-9
N_AUG = 8


def _load_module(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def moo_utils():
    return _load_module("ppo_moo_utils", MOO_DIR / "utils.py")


def datagen():
    return _load_module("ppo_datagen", ROOT / "data" / "data.py")


# ====================================================================
# Cấu hình: mục `ppo` trong rl/config.yaml (train / solve)
# ====================================================================
_PATH_KEYS = ("out", "init", "ckpt", "data_dir")


def load_config(section, path=None):
    import yaml
    path = Path(path or CONFIG_PATH)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        cfg = ((yaml.safe_load(f) or {}).get("ppo") or {}).get(section) or {}
    for k in _PATH_KEYS:
        v = cfg.get(k)
        if isinstance(v, str) and v.lower() != "none" and not Path(v).is_absolute():
            cfg[k] = str(RL_DIR / v)
    return cfg


def rl_path(p):
    """Đường dẫn tương đối -> tính từ thư mục rl/ (None giữ nguyên)."""
    if p is None or (isinstance(p, str) and p.lower() == "none"):
        return None
    p = Path(p)
    return p if p.is_absolute() else RL_DIR / p


def set_seed(seed):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ====================================================================
# Instance: Problem + mảng numpy + đặc trưng tĩnh của nút
# ====================================================================
STATIC_DIM = 12


class Instance:
    """Bọc Problem cho môi trường PPO. Nút: 0 = kho, 1..R = đường, R+1..R+I = thông tin, end = kho cuối."""

    def __init__(self, path):
        P = Problem(str(path))
        self.P, self.path = P, Path(path)
        self.n = P.n_nodes
        self.tt = np.array(P.truck_time)                 # giờ
        self.ut = np.array(P.uav_time)
        self.ee = np.array(P.uav_energy)
        self.sv = np.array(P.service)
        self.profit = np.array(P.profit, dtype=float)
        self.is_info = np.zeros(self.n, bool)
        self.is_info[P.info_ids] = True
        self.reach = np.zeros(self.n, bool)
        self.reach[P.reach_info] = True
        self.road = np.array(P.road_ids, dtype=int)
        self.static = self._static()

    def _static(self):
        P = self.P
        xy = np.array([nd.coord for nd in P.nodes], dtype=float)
        depot = xy[0]
        scale = max(1e-6, np.abs(xy - depot).max())
        rel = (xy - depot) / scale                        # [-1, 1], kho ở gốc -> phép D4 quanh kho
        kind = np.zeros((self.n, 4))
        for i, nd in enumerate(P.nodes):
            kind[i, ["depot_start", "road", "info", "depot_end"].index(nd.kind)] = 1.0
        pmax = max(1.0, self.profit.max())
        e1 = np.array([min(P.e_single[i], 2 * P.theta) / P.theta if P.reachable[i] else 2.0 for i in range(self.n)])
        f = np.zeros((self.n, STATIC_DIM - 2))
        f[:, 0:4] = kind
        f[:, 4] = self.profit / pmax
        f[:, 5] = self.sv / max(P.T, 1e-6) * 10.0
        f[:, 6] = e1
        f[:, 7] = self.ut[0] / max(P.T, 1e-6) * 10.0
        f[:, 8] = self.tt[0] / max(P.T, 1e-6)
        f[:, 9] = self.reach.astype(float)
        return rel.astype(np.float32), f.astype(np.float32)

    def static_aug(self, n_aug=N_AUG):
        """[n_aug, n, STATIC_DIM]: 8 phép đối xứng D4 của tọa độ (quanh kho) + đặc trưng bất biến."""
        rel, f = self.static
        out = []
        for a in range(n_aug):
            x, y = rel[:, 0], rel[:, 1]
            if a & 4:
                x, y = y, x
            if a & 1:
                x = -x
            if a & 2:
                y = -y
            out.append(np.concatenate([np.stack([x, y], 1), f], 1))
        return np.stack(out).astype(np.float32)


# ====================================================================
# Dữ liệu huấn luyện: sinh bằng data/data.py với seed KHÔNG trùng bộ test (42-46)
# ====================================================================
def gen_train_data(out_dir, sizes, dists=("R", "C", "RC"), count=10, seed=1000):
    """sizes: list 'S10', 'S20', 'L100', ... -> ghi out_dir/<suite>/N<n>/*.json, trả về danh sách file."""
    dg = datagen()
    cfg = dg.load_config(None)
    energy = dg.derived_energy(cfg)
    mah = cfg["BATTERY_MAH_DEFAULT"]
    files = []
    for spec in sizes:
        suite, n = spec[0], int(spec[1:])
        s_max = cfg["PROFILES"][suite].get("SORTIES_PER_UAV")
        d_out = Path(out_dir) / suite / f"N{n}"
        d_out.mkdir(parents=True, exist_ok=True)
        for dist in dists:
            for i in range(count):
                sd = seed + i
                path = d_out / dg.file_name(suite, sd, n, dist, cfg["DEFAULT_R"], mah, mah)
                if not path.exists():
                    core = dg.build_core(n, dist, sd, suite, "uniform", cfg)
                    inst, _, _ = dg.build_instance(core, n, dist, sd, seed, suite, cfg["DEFAULT_R"], mah, "uniform",
                                                   s_max, cfg, energy)
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(inst, f)
                files.append(path)
    return files


def das_dennis(M):
    return np.array([(i / M, 1.0 - i / M) for i in range(M + 1)])


def hv_norm(points, ref=(1.0, 1.0)):
    """HV 2D (cực tiểu) trên điểm đã chuẩn hóa (o1 = 1 - f1/F1_REF, o2 = f2/F2_REF), cùng cách với moo."""
    pts = sorted(p for p in points if p[0] < ref[0] and p[1] < ref[1])
    hv, best = 0.0, ref[1]
    for x, y in pts:
        if y < best:
            hv += (ref[0] - x) * (best - y)
            best = y
    return hv


def nondominated(items, key):
    """Lọc phần tử không bị trội (cực tiểu 2 chiều theo key(item)), bỏ điểm trùng."""
    items = sorted(items, key=lambda it: (key(it)[0], key(it)[1]))
    out, best = [], math.inf
    for it in items:
        o = key(it)
        if o[1] < best - 1e-12:
            out.append(it)
            best = o[1]
    return out
