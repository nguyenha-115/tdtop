import argparse
import glob
import json
import os
import re
import sys

import matplotlib
import numpy as np

COLORS = {"ga": "#1f77b4", "lns": "#d62728"}
NAMES = {"ga": "GA", "lns": "LNS"}
YLABEL = {"fitness": "Best fitness", "profit": "Best profit", "energy": "Energy (Wh)"}


def find_file(root, algo, filename):
    if not filename.endswith(".json"):
        filename += ".json"
    base = os.path.join(root, algo)
    direct = os.path.join(base, filename)
    if os.path.isfile(direct):
        return direct
    hits = glob.glob(os.path.join(base, "**", filename), recursive=True)
    if not hits:
        sys.exit(f"Không tìm thấy {filename} trong {base}")
    return sorted(hits)[0]


def find_folder_files(root, algo, subfolder):
    folder = os.path.join(root, algo, subfolder)
    if not os.path.isdir(folder):
        sys.exit(f"Không có thư mục {folder}")
    files = sorted(glob.glob(os.path.join(folder, "**", "*.json"), recursive=True))
    if not files:
        sys.exit(f"Thư mục {folder} không có file .json")
    return files


def out_path(out_dir, algo, json_path):
    stem = os.path.splitext(os.path.basename(json_path))[0]
    parts = stem.split("_")
    s = re.sub(r"\d+$", "", parts[0]) or parts[0]  # S042 -> S, L015 -> L, SA-R012 -> SA-R
    n = parts[1] if len(parts) > 1 else "N_unknown"  # vd N10
    d = os.path.join(out_dir, algo, s, n)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, stem + ".png")


def plot_one(json_path, algo, metric, out_dir, show):
    import matplotlib.pyplot as plt

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    hist = data["history"]
    keys = sorted(hist, key=int)
    x = np.array([int(k) for k in keys])
    y = np.array([hist[k][metric] for k in keys], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))

    # LNS: thêm đường lời giải hiện tại (SA) nếu có
    if metric == "fitness" and "current_fitness" in hist[keys[0]]:
        cur = np.array([hist[k]["current_fitness"] for k in keys], dtype=float)
        ax.plot(x, cur, color="gray", alpha=0.35, lw=1, label="Current (SA)")

    ax.plot(x, y, color=COLORS[algo], lw=2, drawstyle="steps-post",
            label=f"Best so far (final = {y[-1]:.4f})")
    imp = np.where(np.diff(y, prepend=y[0] - 1) != 0)[0]
    ax.scatter(x[imp], y[imp], s=14, color=COLORS[algo], zorder=3)

    # GA: đánh dấu restart
    if algo == "ga":
        prev, first = 0, True
        for k in keys:
            r = hist[k].get("restarts", 0)
            if r > prev:
                ax.axvline(int(k), color="orange", ls=":", lw=1,
                           label="Restart" if first else None)
                first = False
            prev = r

    t = data.get("time")
    title = os.path.splitext(os.path.basename(json_path))[0]
    ax.set_title(f"{NAMES[algo]} convergence — {title}" +
                 (f"  (t = {t:.2f}s)" if t is not None else ""))
    ax.set_xlabel("Thế hệ (generation)" if algo == "ga" else "Vòng lặp (iteration)")
    ax.set_ylabel(YLABEL[metric])
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    out = out_path(out_dir, algo, json_path)
    fig.savefig(out, dpi=150)
    print("Saved:", out)
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Vẽ biểu đồ hội tụ GA / LNS")
    ap.add_argument("--algo", required=True, choices=["ga", "lns"])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--filename", help="vd: S042_N10_C_R3.json")
    g.add_argument("--subfolder", help="vd: S/N10 ")
    ap.add_argument("--root", default="./result", help="thư mục gốc kết quả (mặc định /result)")
    ap.add_argument("--out", default="./img", help="thư mục lưu ảnh")
    ap.add_argument("--metric", default="fitness", choices=list(YLABEL))
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    if not a.show:
        matplotlib.use("Agg")

    if a.filename:
        files = [find_file(a.root, a.algo, a.filename)]
    else:
        files = find_folder_files(a.root, a.algo, a.subfolder)

    for p in files:
        try:
            plot_one(p, a.algo, a.metric, a.out, a.show)
        except Exception as e:  # file lỗi thì bỏ qua, chạy tiếp
            print(f"Bỏ qua {p}: {e}")


if __name__ == "__main__":
    main()