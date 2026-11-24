"""Tổng hợp kết quả NSGA-II + policy trong rl/outputs/compare/<cấu hình>/<S|L>/<N..>/*.json.
HV theo nsga2_kl_rl (ref (1.1, 1.1) trên mục tiêu chuẩn hóa). Chạy: python rl/outputs/compare/summarize.py"""
import json
import sys
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent
CFG = [("pure", "NSGA-II thuần"), ("trained", "POMO cũ"), ("pomo_samedata", "POMO train lại"),
       ("ppo_weighted", "PPO weighted")]
SUBS = ["S/N10", "S/N20", "S/N30", "L/N100"]
BASE = sys.argv[1] if len(sys.argv) > 1 else "pomo_samedata"


def load(cfg, sub):
    return {f.name: json.load(open(f, encoding="utf-8")) for f in sorted((D / cfg / sub).glob("*.json"))}


R = {c: {s: load(c, s) for s in SUBS} for c, _ in CFG}
print(f"{'nhóm':7s} " + " | ".join(f"{n:>22s}" for _, n in CFG))
for s in SUBS:
    cells = []
    for c, _ in CFG:
        d = R[c][s]
        if not d:
            cells.append(f"{'-':>22s}")
            continue
        hv = np.mean([r["hv"] for r in d.values()])
        t = np.mean([r["time"] for r in d.values()])
        cells.append(f"HV={hv:.4f} t={t:5.1f}s ({len(d):2d})")
    print(f"{s:7s} " + " | ".join(cells))
print(f"\nSố instance thắng / thua so với {BASE} (cùng instance, HV):")
for c, n in CFG:
    if c == BASE:
        continue
    row = []
    for s in SUBS:
        a, b = R[c][s], R[BASE][s]
        common = sorted(set(a) & set(b))
        if not common:
            row.append(f"{s}: -")
            continue
        w = sum(a[f]["hv"] > b[f]["hv"] + 1e-9 for f in common)
        l = sum(a[f]["hv"] < b[f]["hv"] - 1e-9 for f in common)
        row.append(f"{s}: {w}/{l} (n={len(common)})")
    print(f"  {n:16s} " + "  ".join(row))
errs = {c: sum(r.get("verify_errors", 0) for s in SUBS for r in R[c][s].values()) for c, _ in CFG}
print("\nlỗi verify:", errs)
