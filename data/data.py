"""
Sinh dữ liệu TD-TOP đa xe tải (MILP đa mục tiêu + thuật toán base) theo đặc tả data.md.

Ví dụ:
    python data.py --suite S                      # 15 instance nhỏ (n=10,20,30 x 5)
    python data.py --suite L                      # 90 instance lớn (n=100..1000 x R,C,RC x 3 seed)
    python data.py --suite SA-R                   # 9 instance x R = 2..6
    python data.py --suite SA-B                   # 9 instance x 4 mức pin
    python data.py --suite L --n 200 --dist RC --seed 42 --count 5
    python data.py --write_config                 # ghi config.json mặc định rồi thoát
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys

import numpy as np

# Cấu hình mặc định (Mục 2 của data.md). Nếu có config.json thì ghi đè (gộp sâu).
DEFAULT_CONFIG = {
    "V_TRUCK": 50.0,
    "V_UAV": 72.0,
    "UAV": {
        "P0": 79.86,
        "Pi": 88.66,
        "U_TIP": 120.0,
        "V0": 4.03,
        "D0": 0.6,
        "RHO": 1.225,
        "Q_SOLIDITY": 0.05,
        "A_DISK": 0.503,
        "P_SENS": 5.0,
        "P_COMP": 8.0,
        "P_TX": 2.0,
    },
    "BATTERY_VOLT": 14.76,
    "BATTERY_MAH_DEFAULT": 4276,
    "BATTERY_RESERVE": 0.0,  # hệ số dự trữ an toàn (0 = không trừ)
    "CAPACITY": 100,  # năng lực C: số nút tối đa một đội xe-UAV phục vụ hiệu quả; K = ceil(n / C)
    "DEFAULT_R": 3,
    "VALUE_RANGE": [1, 100],
    "VALUE_NORMAL": {"mean": 50.0, "std": 20.0},
    "SERVICE_RANGE_MIN": [0.0, 3.0],
    "GRID": 1000,
    "CLUSTER_DECAY": 40.0,
    "CLUSTER_RANGE": [3, 8],
    "MIN_ROAD_DIST": 0.1,
    "ROAD_NOISE_RETRIES": 100,
    "PROFILES": {
        "S": {
            "MAX_COORD": 10.0,
            "T_MAX_MIN": 200.0,
            "ROAD_NODES_PER_TRUCK": 4,
            "ROAD_SIGMA": 0.05,
            "SORTIES_PER_UAV": 2,
        },
        "L": {
            "MAX_COORD": 25.0,
            "T_MAX_MIN": 400.0,
            "ROAD_NODES_DIV": 25,
            "ROAD_NODES_RANGE": [5, 15],
            "ROAD_SIGMA": 0.2,
            "SORTIES_PER_UAV": None,
        },
    },
}
DEFAULT_CONFIG["PROFILES"]["SA-R"] = copy.deepcopy(DEFAULT_CONFIG["PROFILES"]["L"])
DEFAULT_CONFIG["PROFILES"]["SA-B"] = copy.deepcopy(DEFAULT_CONFIG["PROFILES"]["L"])

SUITE_BATTERIES = {"SA-B": [1900, 2850, 4276, 5100]}
SUITE_RS = {"SA-R": [2, 3, 4, 5, 6]}


def deep_merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            deep_merge(cfg, json.load(f))
    return cfg


# Mô hình công suất / năng lượng UAV (Mục 2.2, 2.3)
def p_fly(V, u):
    """Công thức (1): công suất bay của UAV cánh quạt (Zeng et al. 2019), V tính m/s."""
    t1 = u["P0"] * (1.0 + 3.0 * V**2 / u["U_TIP"] ** 2)
    inner = math.sqrt(1.0 + V**4 / (4.0 * u["V0"] ** 4)) - V**2 / (2.0 * u["V0"] ** 2)
    t2 = u["Pi"] * math.sqrt(max(inner, 0.0))
    t3 = 0.5 * u["D0"] * u["RHO"] * u["Q_SOLIDITY"] * u["A_DISK"] * V**3
    return t1 + t2 + t3


def derived_energy(cfg):
    u = cfg["UAV"]
    v_ms = cfg["V_UAV"] / 3.6
    pf = p_fly(v_ms, u)
    p_hover = u["Pi"] + u["P0"]
    p_avio = u["P_SENS"] + u["P_COMP"] + u["P_TX"]
    p_srv = p_avio + p_hover
    e_km = pf / cfg["V_UAV"]  # Wh/km = W * (1/v) h
    return {
        "P_AVIO": p_avio,
        "P_FLY_W": pf,
        "P_HOVER_W": p_hover,
        "P_SRV_W": p_srv,
        "E_FLY_WH_PER_KM": e_km,
    }


def theta_wh(mah, cfg):
    return mah / 1000.0 * cfg["BATTERY_VOLT"] * (1.0 - cfg["BATTERY_RESERVE"])


# Mục 3: sinh nút thông tin
def gen_random_points(n, rng, grid, taken):
    pts = []
    while len(pts) < n:
        need = n - len(pts)
        cand = rng.integers(0, grid + 1, size=(max(need * 2, 16), 2))
        for x, y in cand:
            t = (int(x), int(y))
            if t in taken:
                continue  # trùng -> lấy lại điểm đó
            taken.add(t)
            pts.append(t)
            if len(pts) == n:
                break
    return pts


def gen_clustered_points(n, rng_pts, rng_seeds, grid, taken, cfg):
    """Kiểu Uchoa: S hạt giống cụm (S ~ U{3..8}), các nút còn lại lấy bằng rejection sampling
    với xác suất chấp nhận ~ sum_s exp(-d(p, s)/decay)."""
    lo, hi = cfg["CLUSTER_RANGE"]
    S = int(rng_seeds.integers(lo, hi + 1))
    S = max(1, min(S, n))
    seeds = gen_random_points(S, rng_seeds, grid, taken)
    seeds_arr = np.array(seeds, dtype=float)
    pts = list(seeds)
    decay = cfg["CLUSTER_DECAY"]
    while len(pts) < n:
        remaining = n - len(pts)
        B = max(4096, 60 * remaining)
        cand = rng_pts.integers(0, grid + 1, size=(B, 2))
        d = np.linalg.norm(cand[:, None, :] - seeds_arr[None, :, :], axis=2)
        w = np.minimum(1.0, np.exp(-d / decay).sum(axis=1))
        acc = rng_pts.random(B) < w
        for x, y in cand[acc]:
            t = (int(x), int(y))
            if t in taken:
                continue
            taken.add(t)
            pts.append(t)
            if len(pts) == n:
                break
    return pts, S


def gen_coords(n, dist, eff_seed, max_coord, cfg):
    """Trả về (tọa độ km shape (n,2), số cụm S). Luồng seed theo Mục 6."""
    grid = cfg["GRID"]
    scale = max_coord / grid
    taken = set()
    n_clusters = 0
    if dist == "R":
        pts = gen_random_points(n, np.random.default_rng(eff_seed), grid, taken)
    elif dist == "C":
        pts, n_clusters = gen_clustered_points(
            n, np.random.default_rng(eff_seed), np.random.default_rng(eff_seed + 4000),
            grid, taken, cfg)
    elif dist == "RC":
        n_c = n // 2
        pts_c, n_clusters = gen_clustered_points(
            n_c, np.random.default_rng(eff_seed), np.random.default_rng(eff_seed + 4000),
            grid, taken, cfg)
        pts_r = gen_random_points(n - n_c, np.random.default_rng(eff_seed + 500), grid, taken)
        pts = pts_c + pts_r
    else:
        raise ValueError(f"dist không hợp lệ: {dist}")
    coords = np.array(pts, dtype=float) * scale
    return coords, n_clusters


def gen_values(n, eff_seed, dist_name, cfg):
    """Giá trị thông tin p_i: nguyên, trong [1,100]. Luồng seed_i + 1000."""
    rng = np.random.default_rng(eff_seed + 1000)
    lo, hi = cfg["VALUE_RANGE"]
    if dist_name == "uniform":
        return rng.integers(lo, hi + 1, size=n)
    if dist_name == "normal":
        mu, sd = cfg["VALUE_NORMAL"]["mean"], cfg["VALUE_NORMAL"]["std"]
        out = np.empty(n, dtype=int)
        filled = 0
        while filled < n:
            x = np.rint(rng.normal(mu, sd, size=max(2 * (n - filled), 16))).astype(int)
            x = x[(x >= lo) & (x <= hi)][: n - filled]  # chuẩn cắt cụt: bỏ mẫu ngoài khoảng
            out[filled:filled + len(x)] = x
            filled += len(x)
        return out
    raise ValueError(f"value_dist không hợp lệ: {dist_name}")


def gen_service(n, eff_seed, cfg):
    """Thời gian phục vụ s_i ~ U[0,3] phút, làm tròn 4 chữ số. Luồng seed_i + 2000."""
    rng = np.random.default_rng(eff_seed + 2000)
    lo, hi = cfg["SERVICE_RANGE_MIN"]
    return np.round(rng.uniform(lo, hi, size=n), 4)


# Mục 4: kho và nút đường bộ
def road_nodes_per_truck(n, prof):
    if "ROAD_NODES_PER_TRUCK" in prof:
        return int(prof["ROAD_NODES_PER_TRUCK"])
    lo, hi = prof["ROAD_NODES_RANGE"]
    return int(min(hi, max(lo, math.ceil(n / prof["ROAD_NODES_DIV"]))))


def farthest_point_order(coords, depot, k):
    """Duyệt điểm xa nhất; điểm đầu là nút thông tin gần kho nhất => thứ tự có tính lồng nhau."""
    n = len(coords)
    k = min(k, n)
    first = int(np.argmin(np.linalg.norm(coords - depot, axis=1)))
    order = [first]
    mind = np.linalg.norm(coords - coords[first], axis=1)
    while len(order) < k:
        nxt = int(np.argmax(mind))
        order.append(nxt)
        mind = np.minimum(mind, np.linalg.norm(coords - coords[nxt], axis=1))
    return order


def gen_road_nodes(coords, depot, count, sigma, max_coord, eff_seed, cfg):
    """Mỗi nút đường bộ = bản sao nút thông tin (theo thứ tự FPS) + nhiễu Gauss N(0, sigma^2).
    Sinh tuần tự nên K nhỏ là tiền tố của K lớn (tính lồng nhau)."""
    order = farthest_point_order(coords, depot, count)
    rng = np.random.default_rng(eff_seed + 3000)
    min_d = cfg["MIN_ROAD_DIST"]
    placed = []
    for idx in order:
        base = coords[idx]
        cand = None
        for _ in range(cfg["ROAD_NOISE_RETRIES"]):
            cand = np.clip(base + rng.normal(0.0, sigma, size=2), 0.0, max_coord)
            ok = np.linalg.norm(cand - depot) >= min_d
            if ok and placed:
                ok = np.min(np.linalg.norm(np.array(placed) - cand, axis=1)) >= min_d
            if ok:
                break
        placed.append(cand)  # hết số lần thử thì nhận lần cuối
    return np.array(placed)


# Dựng một "lõi" instance (tọa độ, p_i, s_i, nút đường bộ; độc lập với R và θ)
def build_core(n, dist, seed_i, suite, value_dist, cfg):
    prof = cfg["PROFILES"][suite]
    max_coord = prof["MAX_COORD"]
    T = prof["T_MAX_MIN"]
    depot = np.array([max_coord / 2.0, max_coord / 2.0])
    K = max(1, math.ceil(n / cfg["CAPACITY"]))
    m = road_nodes_per_truck(n, prof)
    want_road = K * m
    n_road = min(want_road, n)
    if n_road < want_road:
        print(f"  [cảnh báo] n={n}: chỉ đủ {n_road}/{want_road} nút đường bộ", file=sys.stderr)

    coords, n_clusters = gen_coords(n, dist, seed_i, max_coord, cfg)
    p = gen_values(n, seed_i, value_dist, cfg)
    s = gen_service(n, seed_i, cfg)
    road = gen_road_nodes(coords, depot, n_road, prof["ROAD_SIGMA"], max_coord, seed_i, cfg)
    return dict(coords=coords, p=p, s=s, road=road, n_clusters=n_clusters,
                K=K, m=m, depot=depot, max_coord=max_coord, T=T, prof=prof)


# Mục 8: dựng JSON đầu ra
def r4(x):
    return round(float(x), 4)


def build_instance(core, n, dist, seed_i, seed_base, suite, R, mah, value_dist, s_max, cfg, energy):
    K = core["K"]
    n_road = len(core["road"])
    depot = core["depot"]
    id_end = n_road + 1
    first_info = n_road + 2
    theta = theta_wh(mah, cfg)

    road_nodes = [{"id": i + 1, "coord": [r4(c[0]), r4(c[1])]} for i, c in enumerate(core["road"])]
    info_nodes = [
        {"id": first_info + i, "coord": [r4(c[0]), r4(c[1])],
         "p": int(core["p"][i]), "s_min": float(core["s"][i])}
        for i, c in enumerate(core["coords"])
    ]
    total_value = int(core["p"].sum())

    inst = {
        "Config": {
            "General": {
                "SUITE": suite,
                "NUM_INFO_NODES": n,
                "NUM_ROAD_NODES": n_road,
                "MAX_COORD": core["max_coord"],
                "T_MAX_MIN": core["T"],
                "COORD_DISTRIBUTION": dist,
                "NUM_CLUSTERS": core["n_clusters"] if dist != "R" else 0,
                "SOURCE": "synthetic",
                "SEED": seed_i,
                "SEED_BASE": seed_base,
                "VALUE_DIST": value_dist,
                "CAPACITY": cfg["CAPACITY"],
                "ROAD_NODES_PER_TRUCK": core["m"],
            },
            "Vehicles": {
                "NUM_TRUCKS": K,
                "UAVS_PER_TRUCK": [R] * K,
                "SORTIES_PER_TRUCK": (R * s_max) if s_max else None,
                "V_TRUCK": cfg["V_TRUCK"],
                "V_UAV": cfg["V_UAV"],
                "TRUCK_DISTANCE": "manhattan",
                "UAV_DISTANCE": "euclid",
                "BATTERY_MAH": mah,
                "BATTERY_VOLT": cfg["BATTERY_VOLT"],
                "THETA_WH": r4(theta),
            },
            "Energy": {
                "P0": cfg["UAV"]["P0"], "Pi": cfg["UAV"]["Pi"], "U_TIP": cfg["UAV"]["U_TIP"],
                "V0": cfg["UAV"]["V0"], "D0": cfg["UAV"]["D0"], "RHO": cfg["UAV"]["RHO"],
                "Q_SOLIDITY": cfg["UAV"]["Q_SOLIDITY"], "A_DISK": cfg["UAV"]["A_DISK"],
                "P_SENS": cfg["UAV"]["P_SENS"], "P_COMP": cfg["UAV"]["P_COMP"],
                "P_TX": cfg["UAV"]["P_TX"],
                "P_AVIO": r4(energy["P_AVIO"]),
                "P_FLY_W": r4(energy["P_FLY_W"]),
                "P_HOVER_W": r4(energy["P_HOVER_W"]),
                "P_SRV_W": r4(energy["P_SRV_W"]),
                "E_FLY_WH_PER_KM": r4(energy["E_FLY_WH_PER_KM"]),
            },
        },
        "Depot": {"id_start": 0, "id_end": id_end, "coord": [r4(depot[0]), r4(depot[1])]},
        "RoadNodes": road_nodes,
        "InfoNodes": info_nodes,
        "Stats": {"TOTAL_VALUE": total_value},
    }

    return inst, theta, total_value


# Kế hoạch sinh theo bộ 
def make_plan(args):
    """Trả về danh sách lõi: (n, dist, seed_i)."""
    suite = args.suite
    base = args.seed
    plan = []
    if suite == "S":
        ns = args.n or [10, 20, 30]
        dists = args.dist or ["R"]
        count = args.count if args.count is not None else 5
        for n in ns:
            for d in dists:
                for k in range(count):
                    plan.append((n, d, base + k))
    elif suite == "L":
        ns = args.n or list(range(100, 1001, 100))
        dists = args.dist or ["R", "C", "RC"]
        count = args.count if args.count is not None else 5
        for n in ns:
            for d in dists:
                for k in range(count):
                    plan.append((n, d, base + k))
    elif suite == "SA-R":
        ns = args.n or list(range(200, 1001, 100))
        dists = args.dist or ["R", "C", "RC"]  # xen kẽ theo thứ tự instance
        count = args.count if args.count is not None else 1
        for j, n in enumerate(ns):
            for k in range(count):
                plan.append((n, dists[j % len(dists)], base + j * count + k))
    elif suite == "SA-B":
        ns = args.n or [300]
        dists = args.dist or ["R", "C", "RC"]
        count = args.count if args.count is not None else 3
        for n in ns:
            for d in dists:
                for k in range(count):
                    plan.append((n, d, base + k))
    return plan


def file_name(suite, seed_i, n, dist, R, mah, default_mah):
    prefix = suite.replace("-", "")  
    name = f"{prefix}{seed_i:03d}_N{n}_{dist}_R{R}"
    if mah != default_mah:  # tránh trùng tên giữa các mức pin của bộ SA-B
        name += f"_B{mah}"
    return name + ".json"


MANIFEST_COLS = ["file", "suite", "n", "dist", "K", "R", "mAh", "theta_wh", "T_min", "seed",
                 "source", "TOTAL_VALUE", "sha256"]


def update_manifest(out_root, rows):
    path = os.path.join(out_root, "manifest.csv")
    existing = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                existing[r["file"]] = r
    for r in rows:
        existing[r["file"]] = {k: r[k] for k in MANIFEST_COLS}
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        for k in sorted(existing):
            w.writerow(existing[k])
    return path


def main():
    ap = argparse.ArgumentParser(description="Sinh dữ liệu TD-TOP đa xe tải")
    ap.add_argument("--suite", choices=["S", "L", "SA-R", "SA-B"], default="L")
    ap.add_argument("--n", type=int, nargs="+", default=None, help="số nút thông tin (theo bộ)")
    ap.add_argument("--dist", nargs="+", choices=["R", "C", "RC"], default=None)
    ap.add_argument("--seed", type=int, default=42, help="seed gốc; instance i dùng seed+i")
    ap.add_argument("--count", type=int, default=None,
                    help="số instance mỗi cấu hình (mặc định theo bảng Mục 1: S=5, L=5, SA-R=1, SA-B=3)")
    ap.add_argument("--R", type=int, nargs="+", default=None, help="UAV mỗi xe")
    ap.add_argument("--battery_mah", type=int, nargs="+", default=None)
    ap.add_argument("--C", type=int, default=None, help="năng lực C (ghi đè config); K = ceil(n/C)")
    ap.add_argument("--sorties_per_uav", type=int, default=None, help="s_max cho |Q_k| = R * s_max")
    ap.add_argument("--value_dist", choices=["uniform", "normal"], default="uniform")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--out", default="data")
    ap.add_argument("--write_config", action="store_true", help="ghi config.json mặc định rồi thoát")
    args = ap.parse_args()

    if args.write_config:
        if os.path.exists(args.config):
            print(f"{args.config} đã tồn tại, không ghi đè.")
        else:
            with open(args.config, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
            print(f"Đã ghi {args.config}")
        return

    cfg = load_config(args.config)
    if args.C is not None:
        cfg["CAPACITY"] = args.C
    energy = derived_energy(cfg)
    default_mah = cfg["BATTERY_MAH_DEFAULT"]

    Rs = args.R or SUITE_RS.get(args.suite, [cfg["DEFAULT_R"]])
    batteries = args.battery_mah or SUITE_BATTERIES.get(args.suite, [default_mah])
    s_max = args.sorties_per_uav or cfg["PROFILES"][args.suite].get("SORTIES_PER_UAV")
    print(f"P_fly={energy['P_FLY_W']:.2f} W, P_hover={energy['P_HOVER_W']:.2f} W, "
          f"P_srv={energy['P_SRV_W']:.2f} W, e_fly={energy['E_FLY_WH_PER_KM']:.4f} Wh/km")

    out_root = args.out
    manifest_rows = []
    for (n, dist, seed_i) in make_plan(args):
        core = build_core(n, dist, seed_i, args.suite, args.value_dist, cfg)
        out_dir = os.path.join(out_root, args.suite, f"N{n}")
        os.makedirs(out_dir, exist_ok=True)
        for R in Rs:
            for mah in batteries:
                inst, theta, total_value = build_instance(
                    core, n, dist, seed_i, args.seed, args.suite, R, mah, args.value_dist,
                    s_max, cfg, energy)
                fname = file_name(args.suite, seed_i, n, dist, R, mah, default_mah)
                fpath = os.path.join(out_dir, fname)
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(inst, f, indent=2)
                with open(fpath, "rb") as f:
                    digest = hashlib.sha256(f.read()).hexdigest()[:16]
                manifest_rows.append({
                    "file": os.path.relpath(fpath, out_root).replace(os.sep, "/"),
                    "suite": args.suite, "n": n, "dist": dist, "K": core["K"], "R": R,
                    "mAh": mah, "theta_wh": r4(theta), "T_min": core["T"], "seed": seed_i,
                    "source": "synthetic", "TOTAL_VALUE": total_value,
                    "sha256": digest,
                })
                print(f"Generated: {fpath}  (K={core['K']}, road={len(core['road'])})")

    if manifest_rows:
        print("Manifest:", update_manifest(out_root, manifest_rows))


if __name__ == "__main__":
    main()