"""So sánh NSGA-II + policy (rl/outputs/compare/<cấu hình>) với NSGA-II của moo (moo/result/nsga2).
HV tính lại từ mặt Pareto lưu trong file, cùng một cách cho mọi cấu hình: mục tiêu chuẩn hóa theo moo
(o1 = 1 - f1/F1_REF, o2 = f2/F2_REF), điểm tham chiếu (1, 1) như moo.
Chạy: python rl/outputs/compare/vs_moo_nsga2.py"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "moo"))
from problem import Problem  # noqa: E402

D = Path(__file__).resolve().parent
MOO = ROOT / "moo" / "result" / "nsga2"
CFG = [("moo_nsga2", "moo NSGA-II"), ("moo_pfgmoea", "moo PFG-MOEA"),("pure", "NSGA-II thuần (rl)"), ("pomo_samedata", "POMO train lại"),
       ("ppo_weighted", "PPO weighted"), ("ppo_o5_offline", "PPO offline O1-O5")]
SUBS = ["S/N10", "S/N20", "S/N30", "L/N100"]


def hv(pts, ref=(1.0, 1.0)):
    pts = sorted(p for p in pts if p[0] < ref[0] and p[1] < ref[1])
    out, best = 0.0, ref[1]
    for x, y in pts:
        if y < best:
            out += (ref[0] - x) * (best - y)
            best = y
    return out


def front(cfg, sub, name):
    if cfg.startswith("moo_"):
        d = json.load(open(MOO.parent / cfg[4:] / sub / name, encoding="utf-8"))
        return [(p["f1_total_value"], p["f2_total_energy_wh"]) for p in d["pareto_front"]], d["time"], d["hypervolume"]
    d = json.load(open(D / cfg / sub / name, encoding="utf-8"))
    return [(p["f1"], p["f2"]) for p in d["front"]], d["time"], None


rows = []
for sub in SUBS:
    for f in sorted((MOO / sub).glob("*.json")):
        P = Problem(str(ROOT / "data" / "data" / sub / f.name))
        r = {"sub": sub, "file": f.name}
        for c, _ in CFG:
            src = MOO.parent / c[4:] if c.startswith("moo_") else D / c
            if not (src / sub / f.name).exists():
                continue
            F, t, hv_file = front(c, sub, f.name)
            r[c] = hv([P.normalize(a, b) for a, b in F])
            r[c + "_t"] = t
            r[c + "_n"] = len(F)
            if hv_file is not None and abs(hv_file - r[c]) > 1e-6:
                print(f"! HV tính lại lệch file moo: {f.name} {hv_file:.6f} vs {r[c]:.6f}")
        rows.append(r)

print("HV (ref (1,1), chuẩn hóa moo) - trung bình 15 instance; t = thời gian TB (s); |F| = số điểm Pareto TB")
print(f"{'nhóm':7s} " + " | ".join(f"{n:>30s}" for _, n in CFG))
for sub in SUBS:
    R = [r for r in rows if r["sub"] == sub]
    cells = []
    for c, _ in CFG:
        Q = [r for r in R if c in r]
        cells.append(f"{np.mean([r[c] for r in Q]):.4f} t={np.mean([r[c + '_t'] for r in Q]):5.1f} |F|={np.mean([r[c + '_n'] for r in Q]):5.1f}"
                     if Q else f"{'-':>30s}")
    print(f"{sub:7s} " + " | ".join(f"{x:>30s}" for x in cells))
print("\nThắng/thua so với moo NSGA-II (HV cùng instance):")
for c, n in CFG[1:]:
    parts = []
    for sub in SUBS:
        Q = [r for r in rows if r["sub"] == sub and c in r]
        w = sum(r[c] > r["moo_nsga2"] + 1e-9 for r in Q)
        l = sum(r[c] < r["moo_nsga2"] - 1e-9 for r in Q)
        gain = 100 * np.mean([(r[c] - r["moo_nsga2"]) / r["moo_nsga2"] for r in Q])
        parts.append(f"{sub}: {w}/{l} ({gain:+.1f}%)")
    print(f"  {n:20s} " + "  ".join(parts))
json.dump(rows, open(D / "vs_moo_nsga2.json", "w"), indent=1)
