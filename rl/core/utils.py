import importlib.util
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[2]
MOO_DIR = ROOT / "moo"
DATA_DIR = ROOT / "data"
DATA_ROOT = DATA_DIR / "data"
if str(MOO_DIR) not in sys.path:
    sys.path.insert(0, str(MOO_DIR))

import problem as _moo_problem  # noqa: E402  (moo/problem.py)
from problem import Problem, State, sortie_metrics  # noqa: E402

EPS = 1e-10          # dung sai trong mask
N_AUG = 8            # nhóm dihedral D4
NODE_DIM, GLOB_DIM, EDGE_DIM, TRUCK_DIM = 13, 11, 19, 4


RL_DIR = ROOT / "rl"
CONFIG_PATH = RL_DIR / "config.yaml"
_PATH_KEYS = ("out", "init", "ckpt")


def config_path_from_argv(argv=None):
    """Lấy giá trị --config (nếu có) từ dòng lệnh, mặc định rl/config.yaml."""
    import argparse
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    return pre.parse_known_args(argv)[0].config or str(CONFIG_PATH)


def load_config(section, path=None):
    """Đọc mục `section` ('train' | 'solve') của file yaml. Đường dẫn tương đối tính từ thư mục rl/."""
    import yaml
    path = Path(path or config_path_from_argv())
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        cfg = (yaml.safe_load(f) or {}).get(section) or {}
    for k in _PATH_KEYS:
        v = cfg.get(k)
        if isinstance(v, str) and v.lower() != "none" and not Path(v).is_absolute():
            cfg[k] = str(RL_DIR / v)
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def moo_utils():
    """Nạp moo/utils.py (verify_solution, to_solution, ...) dưới tên riêng để khỏi đụng tên `utils`."""
    if "moo_utils_mod" not in sys.modules:
        spec = importlib.util.spec_from_file_location("moo_utils_mod", MOO_DIR / "utils.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["moo_utils_mod"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["moo_utils_mod"]


# ====================================================================
# Sinh instance (mô phỏng phân phối của data/data.py)
# ====================================================================
_DATAGEN = None


def _datagen():
    global _DATAGEN
    if _DATAGEN is None:
        spec = importlib.util.spec_from_file_location("tdtop_datagen", DATA_DIR / "data.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _DATAGEN = mod
    return _DATAGEN


VALUE_SCENARIOS = ("uniform", "normal")  # phân phối cơ sở của p_i, cắt về [1, 100] (xem data.py: gen_values)


def generate_problem(n, rng, dist=None, R=None, mah=None, capacity=None, m_road=None, value_dist=None,
                     seed_i=None) -> Problem:
    """Instance ngẫu nhiên với n nút thông tin (tọa độ R/C/RC, p_i, s_i, pin, R_k, K như bộ dữ liệu gốc).
    m_road: số nút đường bộ mỗi xe (None = công thức của bộ dữ liệu); value_dist: uniform | normal."""
    g = _datagen()
    cfg = g.load_config(None)
    dist = dist or rng.choice(["R", "C", "RC"])
    R = R or int(rng.integers(2, 6))
    mah = mah or int(rng.choice([1900, 2850, 4276, 5100]))
    value_dist = value_dist or str(rng.choice(VALUE_SCENARIOS))
    cfg["CAPACITY"] = int(capacity or max(5, int(np.ceil(n / rng.integers(1, 4)))))
    suite = "S" if n <= 30 else "L"
    if m_road is not None:
        cfg["PROFILES"][suite]["ROAD_NODES_PER_TRUCK"] = int(m_road)
    if seed_i is None:
        seed_i = int(rng.integers(0, 10 ** 8))
    core = g.build_core(n, dist, seed_i, suite, value_dist, cfg)
    inst, _, _ = g.build_instance(core, n, dist, seed_i, seed_i, suite, R, mah, value_dist,
                                  cfg["PROFILES"][suite].get("SORTIES_PER_UAV"), cfg, g.derived_energy(cfg))
    import json
    fd, path = tempfile.mkstemp(suffix=".json", prefix="tdtop_gen_")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(inst, f)
        P = Problem(path, 0)
    finally:
        os.remove(path)
        _moo_problem._PROBLEM_CACHE.pop((path, 0), None)
    return P


def make_train_specs(num=1000, seed=2024, n_range=(50, 200), m_range=(5, 15)):
    """Training set cố định (tái lập được): `num` instance, mỗi instance gồm 1 kho, m_r ~ U{5..15} nút đường bộ
    (1 đội xe - UAV, CAPACITY >= n nên K = 1), n_u ~ U{50..200} nút thông tin trong vùng 25 x 25 km,
    s_i ~ U[0, 3] phút, p_i theo kịch bản (uniform / normal) cắt về [1, 100]. Trả về list spec; dựng bằng build_train_problem."""
    rng = np.random.default_rng(seed)
    specs = []
    for i in range(num):
        specs.append(dict(n=int(rng.integers(n_range[0], n_range[1] + 1)),
                          m_road=int(rng.integers(m_range[0], m_range[1] + 1)),
                          dist=str(rng.choice(["R", "C", "RC"])),
                          value_dist=VALUE_SCENARIOS[i % len(VALUE_SCENARIOS)],
                          R=int(rng.integers(2, 6)),
                          mah=int(rng.choice([1900, 2850, 4276, 5100])),
                          seed_i=int(rng.integers(10 ** 6, 10 ** 8))))
    return specs


def build_train_problem(spec) -> Problem:
    return generate_problem(spec["n"], None, dist=spec["dist"], R=spec["R"], mah=spec["mah"],
                            capacity=max(spec["n"], 100), m_road=spec["m_road"],
                            value_dist=spec["value_dist"], seed_i=spec["seed_i"])


def load_problem(path, s_max_default=0) -> Problem:
    return Problem(str(path), s_max_default)


# ====================================================================
# Đặc trưng instance cho mạng (Mục 3.1) + tăng cường D4 (Mục 3.7)
# ====================================================================
_D4 = [lambda x, y: (x, y), lambda x, y: (1 - x, y), lambda x, y: (x, 1 - y), lambda x, y: (1 - x, 1 - y),
       lambda x, y: (y, x), lambda x, y: (1 - y, x), lambda x, y: (y, 1 - x), lambda x, y: (1 - y, 1 - x)]


class Inst:
    """Gói Problem + các mảng numpy/tensor dùng chung cho env và mạng."""

    def __init__(self, P: Problem):
        self.P = P
        n = P.n_nodes
        self.n, self.end, self.K = n, P.end, P.num_trucks
        self.road = np.array(P.road_ids, dtype=np.int64)
        self.info = np.array(P.info_ids, dtype=np.int64)
        self.num_road, self.num_info = len(self.road), len(self.info)
        self.road_pos = -np.ones(n, dtype=np.int64)
        self.road_pos[self.road] = np.arange(self.num_road)
        self.info_pos = -np.ones(n, dtype=np.int64)
        self.info_pos[self.info] = np.arange(self.num_info)
        self.tG = np.array(P.truck_time)
        self.tU = np.array(P.uav_time)
        self.eU = np.array(P.uav_energy)
        self.profit = np.array(P.profit, dtype=np.float64)
        self.service = np.array(P.service, dtype=np.float64)
        self.p_info = self.profit[self.info]
        self.s_info = self.service[self.info]
        self.reach_info = np.array([P.reachable[i] for i in P.info_ids], dtype=bool)
        self.T, self.theta = P.T, P.theta
        self.p_srv, self.p_hover = P.p_srv, P.p_hover
        self.R, self.cap = list(P.R), list(P.cap)
        # LB của Pha A cho xe cuối: TT[j,u] = t_ju + t_u0'
        self.TT = self.tG[np.ix_(self.road, self.road)] + self.tG[self.road, self.end][None, :]
        self.t_to_end = self.tG[self.road, self.end]
        self._build_tensors()

    # ---- đặc trưng nút / cạnh / toàn cục / xe
    def _build_tensors(self):
        P, n = self.P, self.n
        coords = np.array([nd.coord for nd in P.nodes], dtype=float)
        lo, hi = coords.min(0), coords.max(0)
        L = max(float((hi - lo).max()), 1e-6)
        self.coord_norm = (coords - (lo + hi) / 2) / L + 0.5
        kind = {"depot_start": 0, "road": 1, "info": 2, "depot_end": 3}
        self.types = np.array([kind[nd.kind] for nd in P.nodes], dtype=np.int64)
        T, th = self.T, self.theta
        x = np.zeros((n, NODE_DIM), dtype=np.float32)
        x[self.road, 2] = self.tG[0, self.road] / T
        x[self.road, 3] = self.tG[self.road, self.end] / T
        maxp = max(1.0, float(self.p_info.max())) if self.num_info else 1.0
        x[self.info, 4] = self.p_info / maxp
        x[self.info, 5] = self.s_info / T
        x[self.info, 6] = self.p_srv * self.s_info / th
        v1 = np.concatenate([[0], self.road, [self.end]])
        euc = np.array(P.uav_dist)
        x[self.info, 7] = euc[np.ix_(v1, self.info)].min(0) / L
        es = np.array([P.e_single[i] for i in self.info])
        x[self.info, 8] = np.minimum(np.where(np.isfinite(es), es / th, 2.0), 2.0)
        x[np.arange(n), 9 + self.types] = 1.0
        self.node_feat = x
        # cạnh
        ev1 = np.zeros(n, dtype=bool)
        ev1[v1] = True
        E = np.zeros((n, n, EDGE_DIM), dtype=np.float32)
        E[..., 0] = self.tG / T * (ev1[:, None] & ev1[None, :])
        E[..., 1] = self.tU / T
        E[..., 2] = self.eU / th
        E[..., 3:] = np.eye(16, dtype=np.float32)[self.types[:, None] * 4 + self.types[None, :]]
        self.edge_t = torch.from_numpy(E)
        self.types_t = torch.from_numpy(self.types)
        # toàn cục
        c_fly = P.e_fly_per_km
        smax = np.mean([min(s, 10) for s in P.s_max]) / 10.0
        g = [self.K / 5, np.mean(self.R) / 5, max(self.R) / 5, th / 100, T / 10, P.v_truck / 100,
             P.v_uav / 100, c_fly / 5, self.p_srv / 200, self.p_hover / 200, smax]
        self.glob_t = torch.tensor(g, dtype=torch.float32)
        # xe
        maxR = max(1, max(self.R))
        maxcap = max(1, max(self.cap))
        maxs = max(1, max(min(s, 10) for s in P.s_max))
        xt = [[self.R[k] / maxR, min(P.s_max[k], 10) / maxs, self.cap[k] / maxcap, (k + 1) / self.K]
              for k in range(self.K)]
        self.truck_t = torch.tensor(xt, dtype=torch.float32)
        self._aug_cache = None

    def node_tensor(self, n_aug=N_AUG):
        """[A, n, NODE_DIM]: A phiên bản D4 (chỉ tọa độ thay đổi; mọi đặc trưng khác bất biến)."""
        if self._aug_cache is None or self._aug_cache.shape[0] != n_aug:
            out = np.repeat(self.node_feat[None], n_aug, axis=0)
            x, y = self.coord_norm[:, 0], self.coord_norm[:, 1]
            for a in range(n_aug):
                xa, ya = _D4[a](x, y)
                out[a, :, 0], out[a, :, 1] = xa, ya
            self._aug_cache = torch.from_numpy(out)
        return self._aug_cache


# ====================================================================
# Vô hướng hóa (Mục 4)  -  G càng lớn càng tốt
# ====================================================================
def normalize(P, f1, f2):
    return 1.0 - np.asarray(f1) / P.f1_ref, np.asarray(f2) / P.f2_ref


def scalarize(P, f1, f2, lam, kind="tcheby", eps=1e-3):
    """Tchebycheff (mặc định, z* = (-eps,-eps)) hoặc tổng có trọng số. f1,f2 có thể là mảng; lam: (2,) hoặc (...,2)."""
    o1, o2 = normalize(P, f1, f2)
    lam = np.asarray(lam, dtype=float)
    l1, l2 = lam[..., 0], lam[..., 1]
    if kind == "tcheby":
        return -np.maximum(l1 * (o1 + eps), l2 * (o2 + eps))
    return l1 * (1.0 - o1) - l2 * o2


def das_dennis(M=100):
    return np.array([(i / M, 1 - i / M) for i in range(M + 1)])


# ====================================================================
# Pareto (cực tiểu hóa)
# ====================================================================
def dominates_matrix(F):
    A, B = F[:, None, :], F[None, :, :]
    return (A <= B).all(2) & (A < B).any(2)       # D[i,j]: i trội j


def nd_mask(F):
    F = np.asarray(F, dtype=float)
    if len(F) == 0:
        return np.zeros(0, dtype=bool)
    return ~dominates_matrix(F).any(0)


def nd_sort(F):
    """Trả về list các front (mảng chỉ số)."""
    F = np.asarray(F, dtype=float)
    if len(F) == 0:
        return []
    D = dominates_matrix(F)
    cnt = D.sum(0)
    cur = np.flatnonzero(cnt == 0)
    fronts = []
    while len(cur):
        fronts.append(cur)
        nxt = []
        for p in cur:
            qs = np.flatnonzero(D[p])
            cnt[qs] -= 1
            nxt.extend(qs[cnt[qs] == 0].tolist())
        cur = np.array(nxt, dtype=int)
    return fronts


def crowding(F):
    F = np.asarray(F, dtype=float)
    n = len(F)
    cd = np.zeros(n)
    if n <= 2:
        return np.full(n, 1e9)
    for m in range(F.shape[1]):
        o = np.argsort(F[:, m], kind="stable")
        v = F[o, m]
        cd[o[0]] = cd[o[-1]] = 1e9
        sc = (v[-1] - v[0]) or 1.0
        cd[o[1:-1]] += (v[2:] - v[:-2]) / sc
    return cd


def hv2d(F, ref=(1.1, 1.1)):
    F = np.asarray(F, dtype=float)
    if len(F) == 0:
        return 0.0
    F = F[(F[:, 0] < ref[0]) & (F[:, 1] < ref[1])]
    F = F[np.argsort(F[:, 0], kind="stable")]
    hv, prev = 0.0, ref[1]
    for x, y in F:
        if y < prev:
            hv += (ref[0] - x) * (prev - y)
            prev = y
    return float(hv)


# ====================================================================
# Lời giải = (routes, arcs); arcs: {(k, idx): [seq, ...]} chỉ gồm cung có sortie
# ====================================================================
def build_state(P, routes, arcs):
    """Dựng State (simulator moo). Trả None nếu vi phạm (42),(44),(50),|Q_k|, pin (60)."""
    st = State(P, routes)
    seen = set()
    for (k, idx), seqs in arcs.items():
        if idx >= len(st.tG[k]) or len(seqs) > P.R[k]:
            return None
        tg = st.tG[k][idx]
        a, b = st.nl[k][idx], st.nl[k][idx + 1]
        for seq in seqs:
            if not seq:
                return None
            for x in seq:
                if x in seen or P.nodes[x].kind != "info":
                    return None
                seen.add(x)
            F, E, Ef, Es = sortie_metrics(P, seq, a, b, tg)
            if E > P.theta + 1e-9:
                return None
            st.attach(k, idx, seq, F, E, Ef, Es)
    if any(st.cnt[k] > P.cap[k] for k in range(P.num_trucks)):
        return None
    return st


def signature(routes, arcs):
    return hash((tuple(tuple(r) for r in routes),
                 tuple(sorted((key, tuple(sorted(tuple(s) for s in v))) for key, v in arcs.items() if v))))


class Sol:
    """Một lời giải: routes, arcs, f1, f2, feasible."""
    __slots__ = ("routes", "arcs", "f1", "f2", "feasible", "sig", "obj", "rank", "crowd", "lam", "op")

    def __init__(self, P, routes, arcs, st=None):
        self.routes = [list(r) for r in routes]
        self.arcs = {k: [list(s) for s in v] for k, v in arcs.items() if v}
        st = st or build_state(P, self.routes, self.arcs)
        if st is None:
            self.f1, self.f2, self.feasible = 0, 0.0, False
        else:
            self.f1, self.f2, self.feasible = st.profit, st.energy, st.feasible
        self.sig = signature(self.routes, self.arcs)
        o1, o2 = P.normalize(self.f1, self.f2)
        self.obj = np.array([o1, o2])
        self.rank, self.crowd, self.lam, self.op = 0, 0.0, None, ""

    def copy_plan(self):
        return [list(r) for r in self.routes], {k: [list(s) for s in v] for k, v in self.arcs.items()}


def route_time(inst, r):
    tG = inst.tG
    t, cur = 0.0, 0
    for j in r:
        t += tG[cur, j]
        cur = j
    return t + tG[cur, inst.end]


def assign_lambda(P, objs_f, ref_f, Lam, kind="tcheby"):
    """Mục 6.2: lambda(s) = argmax_lam [G_lam(s) - max_{s' in A} G_lam(s')]. objs_f, ref_f: mảng (m,2) của (f1,f2).
    kind: cách vô hướng hóa mà policy được huấn luyện ("tcheby" | "weighted") để lambda gán khớp nghĩa với policy."""
    Gs = scalarize(P, objs_f[:, 0][:, None], objs_f[:, 1][:, None], Lam[None, :, :], kind)
    Gr = scalarize(P, ref_f[:, 0][:, None], ref_f[:, 1][:, None], Lam[None, :, :], kind).max(0)
    return Lam[np.argmax(Gs - Gr[None, :], axis=1)]
