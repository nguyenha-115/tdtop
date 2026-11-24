"""So sánh các thuật toán đa mục tiêu (moo_algorithm) trên cùng các instance TD-TOP.

Kết quả đọc từ moo/result/<algo>/<subfolder>/<instance>.json (utils.py: build_result). Mặc định so sánh mọi
thuật toán có thư mục kết quả; chỉ xét instance mà TẤT CẢ thuật toán đều có kết quả.

Với mỗi instance:
  * Mặt tham chiếu R = tập không trội của hợp các mặt Pareto (chỉ nghiệm khả thi, không lỗi ràng buộc);
    chuẩn hóa theo [min, max] của hợp: o1 = (f1_max - f1)/range, o2 = (f2 - f2_min)/range (cả hai cực tiểu).
  * Chỉ số:
      HV        hypervolume trong không gian chuẩn hóa chung, điểm tham chiếu (1.1, 1.1)      (lớn tốt)
      HV_file   hypervolume ghi trong file: o1 = 1 - f1/F1_REF, o2 = f2/F2_REF, ref (1, 1)  (lớn tốt)
      IGD       khoảng cách trung bình từ R tới mặt của thuật toán                            (nhỏ tốt)
      GD        khoảng cách trung bình từ mặt của thuật toán tới R                            (nhỏ tốt)
      C         tỉ lệ nghiệm của các thuật toán khác bị thuật toán này trội yếu (coverage)    (lớn tốt)
      contrib   tỉ lệ điểm của R do thuật toán đóng góp                                       (lớn tốt)
      spacing   độ đều phân bố (Schott)                                                       (nhỏ tốt)
      size      số nghiệm khả thi trên mặt Pareto                                             (lớn tốt)
    kèm f1_max, f2_at_f1max (năng lượng của nghiệm f1 lớn nhất), time, evaluations, generations, stop_reason.

Đầu ra:
  moo/result/compare/<tên>.csv          từng instance x thuật toán
  moo/result/compare/<tên>_summary.csv  trung bình theo nhóm (subfolder) và theo loại phân bố (C/R/RC)
  In: bảng trung bình, thứ hạng trung bình, số instance thắng, kiểm định Wilcoxon cặp đôi (HV, IGD).
  --plot : vẽ chồng mặt Pareto + hội tụ HV vào moo/img/compare/<subfolder>/<instance>_compare.png
  --run  : chạy lại các thuật toán qua utils.py (cùng seed / tham số) trước khi so sánh.

Ví dụ (từ thư mục moo):
    python compare.py --subfolder S/N10
    python compare.py --subfolder S/N10 --algo nsga2 moead --plot --x evals
    python compare.py --subfolder S/N10 --algo nsga2 moead mode --run --time-limit 60 --seed 42
    python compare.py --algo nsga2 moead --out all_nsga2_moead.csv
"""
import argparse
import json
import re
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from config import ALGO_NAMES, DATA_ROOT, EPS, MOO_ROOT, RESULT_ROOT

IMG_ROOT = MOO_ROOT / "img"
HV_REF = 1.1
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b",
          "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#393b79"]
# chỉ số dùng để xếp hạng: True = càng lớn càng tốt
METRICS = {"HV": True, "HV_file": True, "IGD": False, "GD": False, "C": True, "contrib": True,
           "spacing": False, "size": True}
REPORT_COLS = list(METRICS) + ["f1_max", "f2_at_f1max", "time", "evaluations"]


# ====================================================================
# Đọc kết quả
# ====================================================================
def available_algos(result_root: Path):
    return sorted(d.name for d in result_root.iterdir() if d.is_dir() and d.name != "compare") \
        if result_root.is_dir() else []


def collect(result_root: Path, algos, subfolder=None, filename=None):
    """Danh sách (subfolder, filename) có kết quả của TẤT CẢ thuật toán."""
    sets = []
    for algo in algos:
        base = result_root / algo
        root = base / subfolder if subfolder else base
        pattern = f"**/{filename}" if filename else "**/*.json"
        found = {(f.parent.relative_to(base).as_posix(), f.name) for f in root.glob(pattern)} if root.is_dir() else set()
        if not found:
            print(f"  [{algo}] không có kết quả trong {root}")
        sets.append(found)
    if not sets:
        return []
    common = set.intersection(*sets)
    missing = set.union(*sets) - common
    if missing:
        print(f"  Bỏ qua {len(missing)} instance thiếu kết quả của ít nhất một thuật toán")
    return sorted(common)


def load(result_root: Path, algo, sub, name):
    with open(result_root / algo / sub / name, encoding="utf-8") as f:
        return json.load(f)


def front_array(res):
    """Mảng (f1, f2) các nghiệm khả thi, không lỗi ràng buộc."""
    pts = [(p["f1_total_value"], p["f2_total_energy_wh"]) for p in res.get("pareto_front", [])
           if p.get("violation_h", 0) <= EPS and not p.get("verify_errors")]
    return np.array(pts, dtype=float).reshape(-1, 2)


def instance_type(name: str) -> str:
    """S042_N10_RC_R3 -> RC (kiểu phân bố tọa độ)."""
    m = re.search(r"_(RC|C|R)_", name)
    return m.group(1) if m else "?"


# ====================================================================
# Chỉ số (không gian cực tiểu hóa đã chuẩn hóa)
# ====================================================================
def nondominated(pts):
    if not len(pts):
        return pts
    pts = np.unique(pts, axis=0)                     # sắp theo o1 tăng, o2 tăng
    keep, best_o2 = [], np.inf
    for p in pts:
        if p[1] < best_o2 - 1e-12:
            keep.append(p)
            best_o2 = p[1]
    return np.array(keep).reshape(-1, 2)


def hv2d(pts, ref=(HV_REF, HV_REF)):
    """HV 2 chiều (cực tiểu), chia cho thể tích hộp tham chiếu để nằm trong [0, 1]."""
    pts = np.array([p for p in pts if p[0] < ref[0] and p[1] < ref[1]]).reshape(-1, 2)
    pts = nondominated(pts)
    hv, prev_y = 0.0, ref[1]
    for x, y in pts:
        hv += (ref[0] - x) * (prev_y - y)
        prev_y = y
    return hv / (ref[0] * ref[1])


def mean_min_dist(A, B):
    """Trung bình khoảng cách Euclid từ mỗi điểm của A tới điểm gần nhất trong B."""
    if not len(A) or not len(B):
        return float("nan")
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    return float(d.min(1).mean())


def coverage(A, B):
    """C(A,B): tỉ lệ điểm của B bị ít nhất một điểm của A trội yếu."""
    if not len(B):
        return float("nan")
    if not len(A):
        return 0.0
    dom = (A[:, None, 0] <= B[None, :, 0] + 1e-12) & (A[:, None, 1] <= B[None, :, 1] + 1e-12)
    return float(dom.any(0).mean())


def spacing(A):
    if len(A) < 3:
        return float("nan")
    d = np.abs(A[:, None, :] - A[None, :, :]).sum(-1)
    np.fill_diagonal(d, np.inf)
    dm = d.min(1)
    return float(np.sqrt(((dm - dm.mean()) ** 2).sum() / (len(A) - 1)))


def compare_instance(results):
    """results: {algo: json}. Trả về list dòng (mỗi thuật toán một dòng)."""
    raw = {a: front_array(r) for a, r in results.items()}
    nonempty = [p for p in raw.values() if len(p)]
    if not nonempty:
        return []
    allp = np.vstack(nonempty)
    f1_max, f1_min = allp[:, 0].max(), allp[:, 0].min()
    f2_min, f2_max = allp[:, 1].min(), allp[:, 1].max()
    r1, r2 = max(f1_max - f1_min, 1e-9), max(f2_max - f2_min, 1e-9)

    def norm(p):
        return np.column_stack([(f1_max - p[:, 0]) / r1, (p[:, 1] - f2_min) / r2]) if len(p) else p.reshape(-1, 2)

    nf = {a: norm(p) for a, p in raw.items()}
    ref_front = nondominated(norm(allp))
    ref_set = {tuple(np.round(p, 9)) for p in ref_front}

    rows = []
    for a, res in results.items():
        A, F = nf[a], raw[a]
        others = [nf[b] for b in nf if b != a and len(nf[b])]
        others = np.vstack(others) if others else np.zeros((0, 2))
        if len(F):
            j = max(range(len(F)), key=lambda i: (F[i, 0], -F[i, 1]))
            f1m, f2m = F[j, 0], F[j, 1]
        else:
            f1m = f2m = float("nan")
        hist = res.get("history", {})
        rows.append({
            "algo": a,
            "HV": hv2d(A),
            "HV_file": res.get("hypervolume", float("nan")),
            "IGD": mean_min_dist(ref_front, A),
            "GD": mean_min_dist(A, ref_front),
            "C": coverage(A, others),
            "contrib": len({tuple(np.round(p, 9)) for p in A} & ref_set) / max(1, len(ref_front)),
            "spacing": spacing(A),
            "size": len(A),
            "f1_max": f1m,
            "f2_at_f1max": f2m,
            "time": res.get("time", float("nan")),
            "evaluations": res.get("evaluations", float("nan")),
            "generations": res.get("num_generations", len(hist)),
            "stop_reason": res.get("stop_reason"),
            "verify_errors": len(res.get("verify_errors", [])),
        })
    return rows


# ====================================================================
# Vẽ
# ====================================================================
def plot_instance(results, sub, name, img_root: Path, x_axis="gen"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    field, xlabel = {"gen": (None, "Thế hệ"), "evals": ("evaluations", "Số lần đánh giá"),
                     "time": ("time", "Thời gian (s)")}[x_axis]
    stem = Path(name).stem
    out = img_root / "compare" / sub
    out.mkdir(parents=True, exist_ok=True)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 5.5))
    for ci, (algo, res) in enumerate(results.items()):
        c = COLORS[ci % len(COLORS)]
        pts = front_array(res)
        if len(pts):
            pts = pts[np.lexsort((-pts[:, 0], pts[:, 1]))]
            ax.plot(pts[:, 1], pts[:, 0], "-", color=c, lw=1, alpha=0.5, drawstyle="steps-post")
            ax.scatter(pts[:, 1], pts[:, 0], s=26, color=c, edgecolors="k", linewidths=0.4, zorder=3,
                       label=f"{algo.upper()} ({len(pts)} nghiệm, t={res.get('time', 0):.1f}s)")
        hist = res.get("history", {})
        keys = sorted(hist, key=int)
        if keys:
            x = [int(k) if field is None else hist[k].get(field) for k in keys]
            bx.plot(x, [hist[k]["hypervolume"] for k in keys], color=c, lw=2,
                    label=f"{algo.upper()} (cuối = {hist[keys[-1]]['hypervolume']:.5f})")
    ax.set_xlabel("f2 - Tổng năng lượng UAV (Wh)  ↓")
    ax.set_ylabel("f1 - Tổng giá trị thông tin  ↑")
    ax.set_title(f"Mặt Pareto - {stem}")
    bx.set_xlabel(xlabel)
    bx.set_ylabel("Hypervolume (o1 = 1 - f1/F1_REF, o2 = f2/F2_REF)")
    bx.set_title(f"Hội tụ HV - {stem}")
    for z in (ax, bx):
        z.grid(alpha=0.3)
        z.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    p = out / f"{stem}_compare.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  ✔ Saved {p}")


# ====================================================================
# Chạy lại thuật toán (qua utils.py)
# ====================================================================
def run_algorithms(args):
    for algo in args.algo:
        if algo not in ALGO_NAMES:
            print(f"  '{algo}' không có trong config.ALGO_DEFAULT, bỏ qua --run")
            continue
        cmd = [sys.executable, str(MOO_ROOT / "utils.py"), "--algo", algo, "--seed", str(args.seed),
               "--data-root", args.data_root]
        for flag, val in (("--filename", args.filename), ("--subfolder", args.subfolder),
                          ("--time-limit", args.time_limit), ("--max-gen", args.max_gen),
                          ("--pop-size", args.pop_size), ("--early-stop", args.early_stop),
                          ("--processes", args.processes), ("--ls-rate", args.ls_rate),
                          ("--s-max", args.s_max)):
            if val is not None:
                cmd += [flag, str(val)]
        print(f"\n>>> {' '.join(cmd)}")
        subprocess.run(cmd, cwd=MOO_ROOT, check=False)


# ====================================================================
# Tổng hợp
# ====================================================================
def win_table(df, algos):
    """Số instance mà mỗi thuật toán tốt nhất duy nhất theo từng chỉ số (hòa không tính)."""
    wins = {a: dict.fromkeys(METRICS, 0) for a in algos}
    for m, higher in METRICS.items():
        piv = df.pivot_table(index=["group", "instance"], columns="algo", values=m).dropna()
        for _, row in piv.iterrows():
            best = row.max() if higher else row.min()
            winners = [a for a in row.index if abs(row[a] - best) <= 1e-9]
            if len(winners) == 1:
                wins[winners[0]][m] += 1
    return pd.DataFrame(wins).T


def rank_table(df):
    """Thứ hạng trung bình (1 = tốt nhất, hòa lấy hạng trung bình) theo từng chỉ số."""
    out = {}
    for m, higher in METRICS.items():
        piv = df.pivot_table(index=["group", "instance"], columns="algo", values=m).dropna()
        if len(piv):
            out[m] = piv.rank(axis=1, ascending=not higher, method="average").mean()
    r = pd.DataFrame(out)
    r["mean_rank"] = r.mean(axis=1)
    return r.sort_values("mean_rank")


def summarize(df: pd.DataFrame, algos, summary_path: Path):
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    line = "=" * 110
    by_group = df.groupby(["group", "algo"])[REPORT_COLS].mean()
    by_type = df.groupby(["type", "algo"])[REPORT_COLS].mean()
    overall = df.groupby("algo")[REPORT_COLS].mean()
    print(f"\n{line}\nTrung bình theo nhóm (HV/C/contrib/size: lớn tốt; IGD/GD/spacing: nhỏ tốt)\n{line}")
    print(by_group.round(4).to_string())
    print("\nTrung bình theo loại phân bố:")
    print(by_type.round(4).to_string())
    print("\nTrung bình toàn bộ:")
    print(overall.round(4).to_string())

    pd.concat({"group": by_group.reset_index(level=0).rename(columns={"group": "key"}),
               "type": by_type.reset_index(level=0).rename(columns={"type": "key"}),
               "all": overall.assign(key="all")}, names=["by"]).reset_index() \
        .to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved {summary_path}")

    print("\nThứ hạng trung bình (1 = tốt nhất):")
    print(rank_table(df).round(3).to_string())
    print("\nSố instance thắng (tốt nhất duy nhất) theo chỉ số:")
    print(win_table(df, algos).to_string())

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("\n(Không có scipy -> bỏ qua kiểm định Wilcoxon)")
        return
    print()
    for m in ("HV", "IGD"):
        piv = df.pivot_table(index=["group", "instance"], columns="algo", values=m).dropna()
        for a, b in combinations(algos, 2):
            if a not in piv or b not in piv or len(piv) < 6:
                continue
            diff = piv[a] - piv[b]
            if np.allclose(diff, 0):
                print(f"Wilcoxon {m}: {a} vs {b}  n={len(piv)}  -> như nhau trên mọi instance")
                continue
            p = wilcoxon(piv[a], piv[b]).pvalue
            better = a if (diff.median() > 0) == METRICS[m] else b
            print(f"Wilcoxon {m}: {a} vs {b}  n={len(piv)}  p={p:.4g}  -> {better} tốt hơn"
                  f"{' (có ý nghĩa, p<0.05)' if p < 0.05 else ' (chưa có ý nghĩa)'}")


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="So sánh các thuật toán MOO cho TD-TOP")
    ap.add_argument("--algo", nargs="+", default=None,
                    help="Các thuật toán (thư mục trong result-root). Mặc định: mọi thư mục có kết quả")
    ap.add_argument("--filename", type=str, default=None)
    ap.add_argument("--subfolder", type=str, default=None)
    ap.add_argument("--result-root", type=str, default=str(RESULT_ROOT))
    ap.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    ap.add_argument("--img-root", type=str, default=str(IMG_ROOT))
    ap.add_argument("--out", type=str, default=None, help="Tên file CSV trong result/compare (mặc định theo subfolder)")
    ap.add_argument("--plot", action="store_true", help="Vẽ chồng mặt Pareto + hội tụ HV cho từng instance")
    ap.add_argument("--x", type=str, default="gen", choices=["gen", "evals", "time"], help="Trục x hình hội tụ HV")
    ap.add_argument("--run", action="store_true", help="Chạy lại các thuật toán (utils.py) trước khi so sánh")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--max-gen", type=int, default=None)
    ap.add_argument("--pop-size", type=int, default=None)
    ap.add_argument("--early-stop", type=int, default=None)
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--ls-rate", type=float, default=None)
    ap.add_argument("--s-max", type=int, default=None)
    args = ap.parse_args(argv)

    root = Path(args.result_root)
    if args.algo is None:
        if args.run:
            ap.error("--run cần chỉ rõ --algo")
        args.algo = available_algos(root)
        print(f"Thuật toán có kết quả: {', '.join(args.algo) or '(không có)'}")
    if len(args.algo) < 2:
        ap.error("Cần ít nhất 2 thuật toán để so sánh (truyền --algo a b ...)")

    if args.run:
        if not (args.filename or args.subfolder):
            ap.error("--run cần --filename và/hoặc --subfolder")
        run_algorithms(args)

    items = collect(root, args.algo, args.subfolder, args.filename)
    if not items:
        print("Không có instance chung để so sánh.")
        return
    print(f"So sánh {', '.join(a.upper() for a in args.algo)} trên {len(items)} instance\n")

    rows = []
    for sub, name in items:
        results = {a: load(root, a, sub, name) for a in args.algo}
        inst_rows = compare_instance(results)
        for r in inst_rows:
            r.update(group=sub, instance=Path(name).stem, type=instance_type(name))
        rows += inst_rows
        if inst_rows:
            line = "  ".join(f"{r['algo']}: HV={r['HV']:.4f} IGD={r['IGD']:.4f} C={r['C']:.2f} n={r['size']}"
                             for r in inst_rows)
            print(f"{sub}/{name}:  {line}")
        if args.plot:
            plot_instance(results, sub, name, Path(args.img_root), args.x)

    if not rows:
        print("Không có nghiệm khả thi nào.")
        return
    df = pd.DataFrame(rows)
    head = ["group", "type", "instance", "algo"]
    df = df[head + [c for c in df.columns if c not in head]]
    out_name = args.out or (f"{args.subfolder.replace('/', '_')}.csv" if args.subfolder else "all.csv")
    out = root / "compare" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nSaved {out}")
    summarize(df, args.algo, out.with_name(f"{out.stem}_summary.csv"))


if __name__ == "__main__":
    main()
