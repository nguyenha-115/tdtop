"""Vẽ chồng đường hội tụ HV của mọi thuật toán: các cấu hình rl (rl/outputs/compare/<cấu hình>) và
các thuật toán moo (moo/result/<algo>) — chỉ ĐỌC kết quả, không sửa gì trong moo.

Thang HV chung = thang của moo: o1 = 1 - f1/F1_REF, o2 = f2/F2_REF, điểm tham chiếu (1, 1).
rl dùng cùng chuẩn hóa nhưng ref (1.1, 1.1); vì mọi mặt đều chứa nghiệm rỗng o = (1, 0) nên
    HV_moo = HV_rl - (1.1 * 1.1 - 1) = HV_rl - 0.21.

Mặc định: mỗi nhóm (S/N10, S/N20, S/N30, L/N100) một ô, vẽ HV trung bình qua các instance mà mọi thuật
toán có mặt trong nhóm đều có kết quả (thuật toán không có kết quả cho nhóm thì bỏ qua ở ô đó).
Với --filename: vẽ HV của đúng instance đó (không lấy trung bình).

Ví dụ (từ thư mục gốc):
    python rl/plot_hv.py
    python rl/plot_hv.py --x time
    python rl/plot_hv.py --subfolder S              # mọi nhóm con của S (S/N10, S/N20, S/N30)
    python rl/plot_hv.py --subfolder S/N10 S/N20 --rl pure ppo_weighted --moo nsga2 pfgmoea
    python rl/plot_hv.py --subfolder S/N10 --filename S042_N10_C_R3.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RL_ROOT = ROOT / "rl" / "outputs" / "compare"
MOO_ROOT = ROOT / "moo" / "result"
OUT_ROOT = RL_ROOT / "img"
SUBS = ["S/N10", "S/N20", "S/N30", "L/N100"]
RL_TO_MOO = 1.1 * 1.1 - 1.0
RL_NAMES = {"pure": "NSGA-II thuần (rl)", "trained": "POMO cũ", "pomo_samedata": "POMO train lại",
            "ppo_weighted": "PPO weighted", "ppo_o5_offline": "PPO offline O1-O5"}
RL_SKIP = {"trained", "pomo_samedata"}   # cấu hình POMO: không vẽ (trừ khi chỉ định qua --rl)
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b",
          "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#393b79", "#637939"]


def list_dirs(root: Path, skip=("compare", "img")):
    return sorted(d.name for d in root.iterdir() if d.is_dir() and d.name not in skip) if root.is_dir() else []


# ====================================================================
# Đọc lịch sử HV -> (x, hv) theo thang moo
# ====================================================================
def rl_curve(path: Path, x_axis):
    d = json.load(open(path, encoding="utf-8"))
    h = sorted(d.get("history", []), key=lambda r: r["gen"])
    x = [r["gen"] if x_axis == "gen" else r["t"] for r in h]
    y = [r["hv"] - RL_TO_MOO for r in h]
    return np.array(x, float), np.array(y, float), d.get("hv", np.nan) - RL_TO_MOO


def moo_curve(path: Path, x_axis):
    d = json.load(open(path, encoding="utf-8"))
    h = d.get("history", {}) or {}
    keys = sorted(h, key=int)
    x = [int(k) if x_axis == "gen" else h[k]["time"] for k in keys]
    y = [h[k]["hypervolume"] for k in keys]
    return np.array(x, float), np.array(y, float), d.get("hypervolume", np.nan)


def algos_for(sub, rl_cfgs, moo_algos):
    """[(nhãn, kiểu, thư mục)] các thuật toán có kết quả cho nhóm sub."""
    out = []
    for c in rl_cfgs:
        if (RL_ROOT / c / sub).is_dir() and any((RL_ROOT / c / sub).glob("*.json")):
            out.append((RL_NAMES.get(c, c), "rl", RL_ROOT / c / sub))
    for a in moo_algos:
        if (MOO_ROOT / a / sub).is_dir() and any((MOO_ROOT / a / sub).glob("*.json")):
            out.append((f"moo {a.upper()}", "moo", MOO_ROOT / a / sub))
    return out


def expand_subfolders(subs, rl_cfgs, moo_algos):
    """'S' -> ['S/N10', 'S/N20', ...]: mọi thư mục con (trong rl hoặc moo) có file .json trực tiếp.
    Nhóm đã chứa .json (vd 'S/N10') giữ nguyên."""
    roots = [RL_ROOT / c for c in rl_cfgs] + [MOO_ROOT / a for a in moo_algos]
    out = []
    for s in subs:
        s = s.strip("/\\").replace("\\", "/")
        found = set()
        for r in roots:
            base = r / s
            if not base.is_dir():
                continue
            if any(base.glob("*.json")):
                found.add(s)
            for f in base.rglob("*.json"):
                found.add(f.parent.relative_to(r).as_posix())
        if not found:
            print(f"  Không tìm thấy kết quả cho --subfolder {s}")

        def key(p):  # S/N10 < S/N20 < S/N100
            return [int(t[1:]) if t[1:].isdigit() else t for t in p.split("/")]
        out += [p for p in sorted(found, key=key) if p not in out]
    return out


def mean_curve(curves, discrete):
    """Trung bình trên lưới x chung; instance đã dừng giữ HV cuối, trước điểm ghi đầu tiên thì bỏ qua."""
    curves = [(x, y) for x, y in curves if len(x)]
    if not curves:
        return None
    lo, hi = min(x[0] for x, _ in curves), max(x[-1] for x, _ in curves)
    grid = np.arange(int(lo), int(hi) + 1, dtype=float) if discrete else np.linspace(lo, hi, 400)
    M = np.full((len(curves), len(grid)), np.nan)
    for i, (x, y) in enumerate(curves):
        idx = np.searchsorted(x, grid, side="right") - 1
        ok = idx >= 0
        M[i, ok] = y[idx[ok]]
    with np.errstate(all="ignore"):
        return grid, np.nanmean(M, axis=0)


# ====================================================================
# Vẽ
# ====================================================================
def draw(ax, sub, algos, color_of, x_axis, filename=None):
    if filename:
        names = [filename] if all((d / filename).exists() for _, _, d in algos) else []
    else:
        sets = [{f.name for f in d.glob("*.json")} for _, _, d in algos]
        names = sorted(set.intersection(*sets)) if sets else []
    if not names:
        ax.set_title(f"{sub}: không có instance chung")
        return
    finals = []
    for label, kind, d in algos:
        reader = rl_curve if kind == "rl" else moo_curve
        data = [reader(d / n, x_axis) for n in names]
        mc = mean_curve([(x, y) for x, y, _ in data], discrete=(x_axis == "gen"))
        if mc is None:
            continue
        final = float(np.nanmean([f for _, _, f in data]))
        finals.append((final, label))
        ax.plot(mc[0], mc[1], color=color_of[label], lw=2 if kind == "rl" else 1.6,
                ls="-" if kind == "rl" else "--", label=f"{label} ({final:.5f})")
    what = Path(filename).stem if filename else f"TB {len(names)} instance"
    ax.set_title(f"Hội tụ HV - {sub} ({what})")
    ax.set_xlabel({"gen": "Thế hệ", "time": "Thời gian (s)"}[x_axis])
    ax.set_ylabel("HV (o1 = 1 - f1/F1_REF, o2 = f2/F2_REF, ref (1,1))")
    ax.grid(alpha=0.3)
    # thu hẹp trục y quanh phần cuối để thấy chênh lệch giữa các thuật toán
    ys = [l.get_ydata() for l in ax.get_lines()]
    if ys:
        start = min(np.nanmin(y[len(y) // 10:]) for y in ys)
        top = max(np.nanmax(y) for y in ys)
        pad = max(top - start, 1e-4) * 0.05
        ax.set_ylim(start - pad, top + pad)
    h, l = ax.get_legend_handles_labels()
    order = np.argsort([-float(s.rsplit("(", 1)[1].rstrip(")")) for s in l])   # xếp theo HV cuối giảm dần
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="lower right", fontsize=7)
    print(f"{sub} ({what}): " + ", ".join(f"{lb}={v:.5f}" for v, lb in sorted(finals, reverse=True)))


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Vẽ chồng hội tụ HV của các thuật toán rl và moo")
    ap.add_argument("--subfolder", nargs="+", default=SUBS,
                    help="Một hoặc nhiều nhóm, vd S/N10 L/N100; thư mục cha (S, L) = mọi nhóm con bên trong. "
                         "Mặc định: S/N10 S/N20 S/N30 L/N100")
    ap.add_argument("--filename", type=str, default=None, help="Chỉ vẽ một instance (không lấy trung bình)")
    ap.add_argument("--rl", nargs="+", default=None, help="Cấu hình trong rl/outputs/compare (mặc định: tất cả trừ POMO)")
    ap.add_argument("--moo", nargs="+", default=None, help="Thuật toán trong moo/result (mặc định: tất cả)")
    ap.add_argument("--x", default="gen", choices=["gen", "time"], help="Trục x")
    ap.add_argument("--out", type=str, default=None, help="File ảnh (mặc định rl/outputs/compare/img/...)")
    args = ap.parse_args(argv)

    rl_cfgs = args.rl or [c for c in list_dirs(RL_ROOT) if c not in RL_SKIP]
    moo_algos = args.moo or list_dirs(MOO_ROOT)
    args.subfolder = expand_subfolders(args.subfolder, rl_cfgs, moo_algos)
    if not args.subfolder:
        print("Không có nhóm nào để vẽ.")
        return
    per_sub = {s: algos_for(s, rl_cfgs, moo_algos) for s in args.subfolder}
    labels = list(dict.fromkeys(lb for a in per_sub.values() for lb, _, _ in a))
    color_of = {lb: COLORS[i % len(COLORS)] for i, lb in enumerate(labels)}   # cùng màu ở mọi ô

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(args.subfolder)
    cols = 1 if n == 1 else 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(9 * cols, 6 * rows), squeeze=False)
    for ax, sub in zip(axes.flat, args.subfolder):
        draw(ax, sub, per_sub[sub], color_of, args.x, args.filename)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()

    if args.out:
        out = Path(args.out)
    else:
        tag = Path(args.filename).stem if args.filename else "_".join(s.replace("/", "") for s in args.subfolder)
        out = OUT_ROOT / f"hv_overlay_{tag}_{args.x}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"✔ Saved {out}")


if __name__ == "__main__":
    main()
