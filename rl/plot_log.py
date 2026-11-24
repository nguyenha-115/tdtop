"""Vẽ đường G, infeasible, ent từ log huấn luyện (có làm mượt).

Chạy (trong thư mục rl/ hoặc bất kỳ đâu):
    python plot_log.py                                   # đọc outputs/logs/train.log -> outputs/logs/train.png
    python plot_log.py path/to/train.log --window 9      # cửa sổ làm mượt 9 điểm log
    python plot_log.py --show                            # mở cửa sổ đồ thị

Mỗi dòng log có dạng:
    [N0] it 100/300  G=-0.55  ent=1.2  gnorm=0.8  infeasible=0.00  t=120s
Mỗi điểm cách nhau `log mỗi 5 vòng`, nên --window 5 ~ trung bình 25 vòng huấn luyện.
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

RL_DIR = Path(__file__).resolve().parent
LINE = re.compile(r"\[N(?P<stage>\d+)\]\s+it\s+(?P<it>\d+)/(?P<n_it>\d+)\s+G=(?P<G>-?[\d.]+(?:e-?\d+)?)"
                  r".*?ent=(?P<ent>-?[\d.]+(?:e-?\d+)?).*?infeasible=(?P<infeasible>[\d.]+)")

# màu theo bảng categorical mặc định (sáng): blue, orange, aqua; chữ dùng màu text, không dùng màu series
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
PANELS = [("G", "G (trung bình quỹ đạo khả thi)", "#2a78d6"),
          ("infeasible", "infeasible (tỉ lệ quỹ đạo hỏng)", "#eb6834"),
          ("ent", "ent (entropy) - giảm từ từ", "#1baf7a")]


def read_text(path):
    """Đọc log với mã hóa tự nhận: PowerShell (`>`, `Tee-Object`) ghi UTF-16 LE có BOM; Git Bash/tee ghi UTF-8."""
    raw = Path(path).read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8-sig", errors="replace")


def parse(path):
    rows = []
    for ln in read_text(path).splitlines():
        m = LINE.search(ln)
        if m:
            rows.append((int(m["stage"]), int(m["it"]), int(m["n_it"]), float(m["G"]),
                         float(m["infeasible"]), float(m["ent"])))
    return rows


def smooth(y, w):
    """Trung bình trượt, giữ nguyên độ dài; ở hai đầu dùng cửa sổ thu hẹp (không kéo giá trị về 0)."""
    y = np.asarray(y, dtype=float)
    if w <= 1 or len(y) < 2:
        return y
    k = np.ones(w)
    return np.convolve(y, k, "same") / np.convolve(np.ones_like(y), k, "same")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", default=str(RL_DIR / "outputs" / "logs" / "train.log"))
    ap.add_argument("--out", default=None, help="file ảnh (mặc định: cùng thư mục, đuôi .png)")
    ap.add_argument("--window", type=int, default=5, help="cửa sổ làm mượt (số điểm log)")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    if not a.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(a.log)
    if not path.exists():
        raise SystemExit(f"không thấy {path}. Tạo log bằng: python core/train.py ... 2>&1 | tee outputs/logs/train.log")
    rows = parse(path)
    if len(rows) < 2:
        raise SystemExit(f"{path}: chưa đủ dòng log dạng '[N..] it ../..  G=...' (cần >= 2)")

    # trục x = số vòng huấn luyện tích lũy (nối các giai đoạn curriculum)
    x, off, stage_start, prev = [], 0, [], None
    for st, it, n_it, *_ in rows:
        if prev is not None and st != prev:
            off += last_n
            stage_start.append(off)
        prev, last_n = st, n_it
        x.append(off + it)
    x = np.array(x)
    data = {k: np.array([r[i] for r in rows]) for k, i in (("G", 3), ("infeasible", 4), ("ent", 5))}

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, facecolor=SURFACE)
    for ax, (key, title, color) in zip(axes, PANELS):
        y, ys = data[key], smooth(data[key], a.window)
        ax.set_facecolor(SURFACE)
        ax.plot(x, y, color=color, alpha=0.28, lw=1)                         # thô (mờ)
        ax.plot(x, ys, color=color, lw=2, solid_capstyle="round")            # đã làm mượt
        ax.plot(x[-1], ys[-1], "o", ms=8, color=color, mec=SURFACE, mew=2)  # điểm cuối, viền cách nền
        ax.annotate(f"{ys[-1]:.3g}", (x[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=10, color=INK)
        for s in stage_start:
            ax.axvline(s, color=INK2, lw=0.8, ls=":")
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=6)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, length=0, labelsize=9)
        ax.margins(x=0.02)
    axes[1].set_ylim(bottom=min(0.0, data["infeasible"].min()))
    axes[-1].set_xlabel("vòng huấn luyện" + (" (đường chấm dọc = đổi giai đoạn)" if stage_start else ""),
                        color=INK2, fontsize=10)
    fig.suptitle(f"{path.name}  -  mờ: giá trị log, đậm: trung bình trượt {a.window} điểm", x=0.01, ha="left",
                 fontsize=11, color=INK2)
    fig.tight_layout(rect=(0, 0, 0.96, 0.97))
    out = Path(a.out) if a.out else path.with_suffix(".png")
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    print(f"đã lưu {out}")

    # tóm tắt bằng chữ: so đầu và cuối (trung bình 1/4 đầu vs 1/4 cuối) để đọc nhanh xu hướng
    q = max(1, len(rows) // 4)
    

if __name__ == "__main__":
    main()
