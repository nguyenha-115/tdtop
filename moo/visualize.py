"""Trực quan hóa kết quả các thuật toán đa mục tiêu (moo_algorithm) cho bài toán TD-TOP (xe tải + UAV).

Đọc kết quả  : moo/result/<algo>/<subfolder>/<instance>.json   (do utils.py ghi: build_result)
Đọc dữ liệu  : data/data/<subfolder>/<instance>.json            (tổng giá trị Σp, nếu có)
Ghi ảnh      : moo/img/<algo>/<subfolder>/<instance>_<loại>.png  (nhiều --algo -> moo/img/compare/...)

Các loại hình (--plot):
    pareto : mặt Pareto f2 (năng lượng UAV, Wh, ↓) - f1 (giá trị thông tin, ↑), đánh dấu nghiệm tiêu biểu.
    hv     : hội tụ hypervolume của kho Pareto + kích thước mặt Pareto (trục x: --x gen | evals | time).

Nghiệm đánh dấu trên mặt Pareto (--solution, có thể nhiều; chỉ khi vẽ 1 thuật toán):
    knee   : điểm "khuỷu" của mặt Pareto (bỏ qua nghiệm rỗng)
    max_f1 : giá trị thông tin lớn nhất (năng lượng nhỏ nhất khi hòa)
    min_f2 : năng lượng nhỏ nhất trong các nghiệm có phục vụ (f1 > 0)
    <số i> : nghiệm thứ i khi sắp mặt Pareto theo f2 tăng (0 = năng lượng nhỏ nhất; -1 = lớn nhất)

Ghi chú: nghiệm f1 = 0, f2 = 0 (không bay UAV) là đầu mút hợp lệ của mặt Pareto; --drop-empty để bỏ khỏi hình.

Cách chạy:
    python visualize.py --filename S042_N10_C_R3.json --subfolder S/N10
    python visualize.py --filename S042_N10_C_R3.json                       # tự tìm subfolder
    python visualize.py --subfolder S/N10 --plot pareto
    python visualize.py --filename S042_N10_C_R3.json --solution knee max_f1 3
    python visualize.py --filename S042_N10_C_R3.json --algo nsga2 moead --x evals   # so sánh pareto + hv
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from config import DATA_ROOT, EPS, MOO_ROOT, RESULT_ROOT  # noqa: E402

IMG_ROOT = MOO_ROOT / "img"
ALGO_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b",
               "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#393b79"]
PLOTS = ["pareto", "hv"]
X_AXES = {"gen": ("Thế hệ (generation)", None),
          "evals": ("Số lần đánh giá (evaluations)", "evaluations"),
          "time": ("Thời gian (s)", "time")}
MARKERS = {"knee": ("red", "Knee"), "max_f1": ("darkgreen", "max f1"), "min_f2": ("purple", "min f2")}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Vẽ kết quả thuật toán đa mục tiêu (TD-TOP)")
    p.add_argument("--filename", type=str, help="Tên file JSON instance, vd S042_N10_R_R3.json")
    p.add_argument("--subfolder", type=str, help="Thư mục con, vd S/N10")
    p.add_argument("--algo", type=str, nargs="+", default=["nsga2"],
                   help="Thư mục kết quả trong result-root (nhiều -> so sánh). Default: nsga2")
    p.add_argument("--plot", type=str, nargs="+", default=PLOTS, choices=PLOTS,
                   help="Loại hình cần vẽ. Default: pareto hv")
    p.add_argument("--solution", type=str, nargs="+", default=["knee", "max_f1"],
                   help="Nghiệm đánh dấu: knee | max_f1 | min_f2 | chỉ số theo f2 tăng")
    p.add_argument("--x", type=str, default="gen", choices=list(X_AXES), help="Trục x của hình hv")
    p.add_argument("--drop-empty", "--drop_empty", action="store_true",
                   help="Bỏ nghiệm f1 = 0 (không phục vụ nút nào) khỏi hình")
    p.add_argument("--result-root", "--result_root", type=str, default=str(RESULT_ROOT))
    p.add_argument("--data-root", "--data_root", type=str, default=str(DATA_ROOT))
    p.add_argument("--img-root", "--img_root", type=str, default=str(IMG_ROOT))
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args(argv)


# ====================================================================
# Đọc file
# ====================================================================
def collect(args):
    """Danh sách (filename, subfolder) cần vẽ, tìm trong kết quả của thuật toán đầu tiên."""
    base = Path(args.result_root) / args.algo[0]
    if args.filename and args.subfolder:
        return [(args.filename, args.subfolder)]
    if args.filename:
        matches = sorted(base.glob(f"**/{args.filename}"))
        if not matches:
            sys.exit(f"Không tìm thấy {args.filename} trong {base}")
        return [(m.name, m.parent.relative_to(base).as_posix()) for m in matches]
    if args.subfolder:
        folder = base / args.subfolder
        if not folder.is_dir():
            sys.exit(f"Không có thư mục kết quả: {folder}")
        return [(f.name, args.subfolder) for f in sorted(folder.glob("*.json"))]
    sys.exit("Cần truyền --filename và/hoặc --subfolder")


def load_json(path: Path):
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def total_profit(args, subfolder, filename):
    """Σp của instance (None nếu không có file dữ liệu)."""
    data = load_json(Path(args.data_root) / subfolder / filename)
    return None if data is None else sum(nd["p"] for nd in data["InfoNodes"])


def out_path(args, algo_dir, subfolder, name, suffix) -> Path:
    p = Path(args.img_root) / algo_dir / subfolder / f"{name}_{suffix}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save(fig, path, dpi):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Saved {path}")


# ====================================================================
# Mặt Pareto & chọn nghiệm
# ====================================================================
def front_of(result, drop_empty=False):
    """Nghiệm khả thi trên mặt Pareto, sắp theo f2 tăng (f1 giảm khi hòa)."""
    front = [e for e in result.get("pareto_front", []) if e.get("violation_h", 0) <= EPS]
    if drop_empty:
        front = [e for e in front if e["f1_total_value"] > 0]
    return sorted(front, key=lambda e: (e["f2_total_energy_wh"], -e["f1_total_value"]))


def xy(front):
    return np.array([[e["f2_total_energy_wh"], e["f1_total_value"]] for e in front], dtype=float).reshape(-1, 2)


def knee_index(front):
    """Điểm khuỷu: nằm phía trên (f1 cao so với f2) và xa nhất đoạn nối hai đầu mút (không gian chuẩn hóa),
    bỏ qua nghiệm rỗng."""
    idx = [i for i, e in enumerate(front) if e["f1_total_value"] > 0] or list(range(len(front)))
    if len(idx) < 3:
        return idx[-1]
    pts = xy([front[i] for i in idx])
    norm = (pts - pts.min(0)) / np.maximum(np.ptp(pts, 0), 1e-12)
    a, b = norm[0], norm[-1]
    u, v = b - a, norm - a
    d = (u[0] * v[:, 1] - u[1] * v[:, 0]) / max(np.linalg.norm(u), 1e-12)   # > 0: phía trên dây cung
    return idx[int(np.argmax(d))]


def pick_solutions(front, wanted):
    """[(nhãn, chỉ số trong front)] theo --solution, bỏ trùng."""
    if not front:
        return []
    out, seen = [], set()
    served = [i for i, e in enumerate(front) if e["f1_total_value"] > 0]
    for s in wanted:
        if s == "knee":
            i = knee_index(front)
        elif s == "max_f1":
            i = max(range(len(front)), key=lambda j: (front[j]["f1_total_value"], -front[j]["f2_total_energy_wh"]))
        elif s == "min_f2":
            i = served[0] if served else 0
        else:
            try:
                i = int(s)
            except ValueError:
                print(f"  --solution '{s}' không hợp lệ, bỏ qua")
                continue
            if not -len(front) <= i < len(front):
                print(f"  --solution {i} ngoài phạm vi (mặt Pareto có {len(front)} nghiệm), bỏ qua")
                continue
            i %= len(front)
            s = f"sol{i}"
        if i not in seen:
            seen.add(i)
            out.append((s, i))
    return out


# ====================================================================
# 1. Mặt Pareto
# ====================================================================
def plot_pareto(args, results, sum_p, filename, subfolder, algo_dir):
    name = Path(filename).stem
    single = len(results) == 1
    fig, ax = plt.subplots(figsize=(9.5, 6))

    f1_ref = None
    for ci, (algo, res) in enumerate(results.items()):
        f1_ref = f1_ref or (res.get("hv_reference") or {}).get("f1_norm")
        front = front_of(res, args.drop_empty)
        if not front:
            continue
        pts = xy(front)
        c = ALGO_COLORS[ci % len(ALGO_COLORS)]
        ax.plot(pts[:, 0], pts[:, 1], "-", color=c, lw=1.2, alpha=0.6, drawstyle="steps-post")
        ax.scatter(pts[:, 0], pts[:, 1], s=30, color=c, edgecolors="k", linewidths=0.5, zorder=3,
                   label=f"{algo.upper()}: {len(front)} nghiệm, HV={res.get('hypervolume', 0):.4f}, "
                         f"{res.get('evaluations', 0)} eval, t={res.get('time', 0):.1f}s")
        if single:
            for lab, i in pick_solutions(front, args.solution):
                col, txt = MARKERS.get(lab, ("orange", lab))
                e = front[i]
                ax.scatter([e["f2_total_energy_wh"]], [e["f1_total_value"]], s=190, facecolors="none",
                           edgecolors=col, linewidths=1.8, zorder=4)
                ax.annotate(f"{txt}\n({e['f2_total_energy_wh']:.1f} Wh, {e['f1_total_value']:.0f})",
                            (e["f2_total_energy_wh"], e["f1_total_value"]), textcoords="offset points",
                            xytext=(10, -22), fontsize=8, color=col)

    x0 = ax.get_xlim()[0]
    if sum_p is not None:
        ax.axhline(sum_p, color="gray", ls=":", lw=1)
        ax.text(x0, sum_p, f" Σp = {sum_p}", va="bottom", fontsize=8, color="gray")
    if f1_ref is not None and (sum_p is None or f1_ref < sum_p - 1e-9):   # có nút UAV không với tới được
        ax.axhline(f1_ref, color="gray", ls="--", lw=0.8)
        ax.text(x0, f1_ref, f" Σp với tới được = {f1_ref:.0f}", va="bottom", fontsize=8, color="gray")
    ax.set_xlabel("f2 - Tổng năng lượng UAV (Wh)  ↓")
    ax.set_ylabel("f1 - Tổng giá trị thông tin  ↑")
    ax.set_title(f"Mặt Pareto - {name}")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, out_path(args, algo_dir, subfolder, name, "pareto"), args.dpi)


# ====================================================================
# 2. Hội tụ hypervolume
# ====================================================================
def history_series(res, x_axis):
    """(x, hv, front_size) theo thế hệ từ res['history']."""
    hist = res.get("history", {})
    keys = sorted(hist, key=int)
    if not keys:
        return None
    field = X_AXES[x_axis][1]
    x = np.array([int(k) if field is None else hist[k].get(field, np.nan) for k in keys], dtype=float)
    hv = np.array([hist[k]["hypervolume"] for k in keys], dtype=float)
    fs = np.array([hist[k]["front_size"] for k in keys], dtype=float)
    return x, hv, fs


def plot_hv(args, results, filename, subfolder, algo_dir):
    name = Path(filename).stem
    xlabel = X_AXES[args.x][0]
    single = len(results) == 1
    if single:
        fig, ax = plt.subplots(figsize=(9.5, 5))
        ax2 = ax.twinx()
    else:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    title = f"Hội tụ hypervolume - {name}"
    for ci, (algo, res) in enumerate(results.items()):
        s = history_series(res, args.x)
        if s is None:
            continue
        x, hv, fs = s
        c = ALGO_COLORS[ci % len(ALGO_COLORS)]
        ax.plot(x, hv, color=c, lw=2, label=f"{algo.upper()} HV (cuối = {hv[-1]:.5f})")
        if single:
            ax2.plot(x, fs, color="gray", lw=1, alpha=0.6, label="Kích thước mặt Pareto")
            if res.get("stop_reason"):
                title += f"\n(dừng: {res['stop_reason']})"
        else:
            ax2.plot(x, fs, color=c, lw=1.5, label=f"{algo.upper()} (cuối = {int(fs[-1])})")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Hypervolume (o1 = 1 - f1/F1_REF, o2 = f2/F2_REF; ref (1, 1))")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax2.set_ylabel("Số nghiệm trên mặt Pareto")
    if single:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=9)
    else:
        ax2.set_xlabel(xlabel)
        ax2.set_title(f"Kích thước mặt Pareto - {name}")
        ax2.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
        ax2.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    save(fig, out_path(args, algo_dir, subfolder, name, "hv" if args.x == "gen" else f"hv_{args.x}"), args.dpi)


# ====================================================================
def run(args, filename, subfolder):
    results = {}
    for algo in args.algo:
        res = load_json(Path(args.result_root) / algo / subfolder / filename)
        if res is None:
            print(f"  Không có kết quả {algo}/{subfolder}/{filename}")
            continue
        results[algo] = res
    if not results:
        return
    algo_dir = args.algo[0] if len(results) == 1 else "compare"
    if "pareto" in args.plot:
        plot_pareto(args, results, total_profit(args, subfolder, filename), filename, subfolder, algo_dir)
    if "hv" in args.plot:
        plot_hv(args, results, filename, subfolder, algo_dir)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    for fname, sfolder in collect(args):
        try:
            run(args, fname, sfolder)
        except Exception as e:  # file lỗi thì bỏ qua, chạy tiếp
            print(f"Bỏ qua {sfolder}/{fname}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
