import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
parser.add_argument(
    "--filename",
    type=str,
    help="Tên file JSON instance, vd S042_N10_R_R3.json",
)
parser.add_argument(
    "--subfolder",
    type=str,
    help="Thư mục con dưới data_root/result_root/algo, vd S/N10",
)
parser.add_argument(
    "--algo",
    type=str,
    default="f1",
    help="Thư mục kết quả trong result_root (ortool.py ghi vào 'f1'). Default: f1",
)
parser.add_argument("--data_root", type=str, default="../data/data", help="Thư mục gốc data (instance JSON)")
parser.add_argument("--result_root", type=str, default="./result", help="Thư mục gốc kết quả")
parser.add_argument("--img_root", type=str, default="./img", help="Thư mục lưu ảnh")

args = parser.parse_args()

DATA_ROOT = args.data_root
RESULT_ROOT = args.result_root
IMG_ROOT = args.img_root

filename = args.filename
subfolder = args.subfolder

files = []  # list[(filename, subfolder)]

if filename and subfolder:
    files.append((filename, subfolder))
elif filename:
    # Tìm file trong kết quả để suy ra subfolder (giống cách ortool.py làm với data)
    matches = glob.glob(os.path.join(RESULT_ROOT, args.algo, "**", filename), recursive=True)
    if not matches:
        print(f"Không tìm thấy {filename} trong {os.path.join(RESULT_ROOT, args.algo)}")
        sys.exit(1)
    for m in matches:
        sub = os.path.relpath(os.path.dirname(m), os.path.join(RESULT_ROOT, args.algo))
        files.append((os.path.basename(m), sub))
elif subfolder:
    result_subfolder_path = os.path.join(RESULT_ROOT, args.algo, subfolder)
    if not os.path.isdir(result_subfolder_path):
        raise FileNotFoundError(f"Result subfolder not found: {result_subfolder_path}")
    for f in sorted(os.listdir(result_subfolder_path)):
        if f.endswith(".json") and os.path.isfile(os.path.join(result_subfolder_path, f)):
            files.append((f, subfolder))
else:
    print("Cần truyền --filename và/hoặc --subfolder")
    sys.exit(1)


def manhattan_path(p, q):
    """Xe tải đi theo khoảng cách Manhattan -> vẽ đường gấp khúc chữ L (ngang rồi dọc)."""
    return [(p[0], p[1]), (q[0], p[1]), (q[0], q[1])]


def run(filename: str, subfolder: str):
    data_path = os.path.join(DATA_ROOT, subfolder, filename)
    result_path = os.path.join(RESULT_ROOT, args.algo, subfolder, filename)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Result file not found: {result_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    # ------------------------------------------------------------------
    # Đọc dữ liệu instance
    # ------------------------------------------------------------------
    depot = data["Depot"]
    id_start = depot["id_start"]
    id_end = depot["id_end"]
    depot_coord = np.array(depot["coord"], dtype=float)

    road_nodes = data["RoadNodes"]
    info_nodes = data["InfoNodes"]

    coord = {id_start: depot_coord, id_end: depot_coord}
    for nd in road_nodes:
        coord[nd["id"]] = np.array(nd["coord"], dtype=float)
    for nd in info_nodes:
        coord[nd["id"]] = np.array(nd["coord"], dtype=float)

    road_ids = [nd["id"] for nd in road_nodes]
    road_set = set(road_ids)
    info_ids = [nd["id"] for nd in info_nodes]
    prize = {nd["id"]: nd["p"] for nd in info_nodes}

    # ------------------------------------------------------------------
    # Đọc kết quả
    # ------------------------------------------------------------------
    routes = result.get("routes", [])
    status = result.get("status", "Unknown")
    objective = result.get("objective")
    run_time = result.get("time_sec", result.get("time"))

    served = set()
    n_sorties = 0
    for rt in routes:
        for s in rt.get("sorties", []):
            served.update(s.get("served_nodes", []))
            n_sorties += 1

    total_prize = sum(prize.get(i, 0) for i in served)
    obj_str = f"{objective:.2f}" if isinstance(objective, (int, float)) else f"{total_prize:.2f}"
    time_str = f"{run_time:.2f}s" if isinstance(run_time, (int, float)) else "N/A"

    # ------------------------------------------------------------------
    # Thiết lập hình vẽ
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_title(
        f"Truck-UAV Solution — {filename}\n"
        f"Status: {status} | Reward: {obj_str} | Trucks used: {len(routes)} | "
        f"Sorties: {n_sorties} | Served: {len(served)}/{len(info_ids)} | Time: {time_str}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.25)

    cmap = plt.get_cmap("tab10")
    sortie_styles = ["--", ":", "-."]
    legend_items = []
    show_route_legend = len(routes) <= 10

    # ------------------------------------------------------------------
    # Vẽ tuyến xe tải (đường gấp khúc Manhattan) và các sortie UAV
    # ------------------------------------------------------------------
    for idx, rt in enumerate(routes):
        color = cmap(idx % 10)
        route = rt["route"]  # [id_start, ..., id_end]

        for a, b in zip(route[:-1], route[1:]):
            pts = np.array(manhattan_path(coord[a], coord[b]))
            ax.plot(pts[:, 0], pts[:, 1], "-", lw=2.2, color=color, alpha=0.9, zorder=2)
            # mũi tên chỉ hướng tại điểm cuối đoạn
            ax.annotate(
                "",
                xy=pts[-1],
                xytext=pts[-2] if not np.allclose(pts[-2], pts[-1]) else pts[0],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0, mutation_scale=14,
                                shrinkA=0, shrinkB=0),
                zorder=2,
            )

        # Sortie UAV: đường thẳng nét đứt cùng màu xe tải
        sorties = rt.get("sorties", [])
        for s_idx, s in enumerate(sorties):
            path = s.get("path", [])
            ls = sortie_styles[s_idx % len(sortie_styles)]
            for a, b in zip(path[:-1], path[1:]):
                ax.annotate(
                    "",
                    xy=coord[b],
                    xytext=coord[a],
                    arrowprops=dict(arrowstyle="->", color='orange', lw=1.3, ls=ls,
                                    alpha=0.85, shrinkA=3, shrinkB=3),
                    zorder=3,
                )
            # Nếu UAV không tìm được nút thu hồi thì land = None -> chỉ vẽ phần có sẵn
            if len(path) >= 2 and len(info_ids) <= 60:
                mid = (coord[path[0]] + coord[path[1]]) / 2.0
                ax.text(mid[0], mid[1], f"S{s.get('id', s_idx + 1)}", fontsize=6,
                        color=color, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

        if show_route_legend:
            legend_items.append(
                Line2D([0], [0], color=color, lw=2.5,
                       label=f"Truck {rt.get('id', idx + 1)} "
                             f"({len(sorties)} sortie, {len(route) - 2} stops)")
            )

    # ------------------------------------------------------------------
    # Nút đường (road nodes): hình thoi cam (giống visualize.py), kích thước lớn hơn
    # ------------------------------------------------------------------
    ROAD_SIZE = 130          
    INFO_MIN, INFO_MAX = 70, 220  
    DEPOT_SIZE = 260

    if road_ids:
        rxy = np.array([coord[i] for i in road_ids])
        ax.scatter(rxy[:, 0], rxy[:, 1], c="orange", marker="D", s=ROAD_SIZE,
                   edgecolors="k", linewidths=0.9, zorder=4)

    # ------------------------------------------------------------------
    # Nút thông tin (info nodes): tròn, viridis theo p, kích thước theo p (giống visualize.py)
    # Đã được phục vụ: viền đen đậm; chưa được phục vụ: mờ hơn
    # ------------------------------------------------------------------
    if info_ids:
        p_arr = np.array([prize[i] for i in info_ids], dtype=float)
        p_min = p_arr.min()
        p_ptp = max(np.ptp(p_arr), 1e-9)
        sizes = INFO_MIN + (INFO_MAX - INFO_MIN) * (p_arr - p_min) / p_ptp
        xy = np.array([coord[i] for i in info_ids])
        is_served = np.array([i in served for i in info_ids])

        sc = None
        if is_served.any():
            sc = ax.scatter(xy[is_served, 0], xy[is_served, 1], c=p_arr[is_served],
                            s=sizes[is_served], cmap="viridis", vmin=p_min,
                            vmax=p_min + p_ptp, marker="o", alpha=0.9,
                            edgecolors="k", linewidths=1.2, zorder=5)
        if (~is_served).any():
            sc2 = ax.scatter(xy[~is_served, 0], xy[~is_served, 1], c=p_arr[~is_served],
                             s=sizes[~is_served], cmap="viridis", vmin=p_min,
                             vmax=p_min + p_ptp, marker="o", alpha=0.4,
                             edgecolors="k", linewidths=0.4, zorder=5)
            sc = sc or sc2
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Giá trị thông tin $p_i$")

    # Depot: hình vuông đỏ (giống visualize.py)
    dx, dy = depot_coord
    ax.scatter([dx], [dy], c="red", marker="s", s=DEPOT_SIZE, edgecolors="k",
               linewidths=1.0, zorder=7)

    # ------------------------------------------------------------------
    # Số hiệu (id) ngay trên đầu mỗi node
    # ------------------------------------------------------------------
    n_all = len(road_ids) + len(info_ids)
    label_fs = 9 if n_all <= 40 else 7 if n_all <= 100 else 5
    for nid in road_ids + info_ids:
        x, y = coord[nid]
        text = str(nid) if nid in road_set else f"{nid} (p={prize[nid]})"
        ax.annotate(text, (x, y), xytext=(0, 10), textcoords="offset points",
                    ha="center", va="bottom", fontsize=label_fs, fontweight="bold",
                    color="darkred" if nid in road_set else "black", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
    ax.annotate(str(id_start), (dx, dy), xytext=(0, 12), textcoords="offset points", ha="center",
                va="bottom", fontsize=label_fs, fontweight="bold", color="red", zorder=9,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))

    ax.set_aspect("equal", adjustable="datalim")

    # ------------------------------------------------------------------
    # Legend (ngoài vùng vẽ)
    # ------------------------------------------------------------------
    base = [
        Line2D([0], [0], marker="s", color="w", label="Depot", markerfacecolor="red",
               markeredgecolor="k", markersize=13),
        Line2D([0], [0], marker="D", color="w", label="Road node", markerfacecolor="orange",
               markeredgecolor="k", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Info node (served)",
               markerfacecolor="mediumseagreen", markeredgecolor="k", markeredgewidth=1.4,
               markersize=11),
        Line2D([0], [0], marker="o", color="w", label="Info node (not served)",
               markerfacecolor="mediumseagreen", markeredgecolor="gray", alpha=0.4,
               markersize=11),
        Line2D([0], [0], color="k", lw=2.2, label="Truck route (Manhattan)"),
        Line2D([0], [0], color="k", lw=1.3, ls="--", label="UAV sortie (Euclid)"),
    ]
    ax.legend(handles=base + legend_items, bbox_to_anchor=(1.15, 1), loc="upper left",
              borderaxespad=0, fontsize=8)

    # ------------------------------------------------------------------
    # Lưu ảnh
    # ------------------------------------------------------------------
    name = os.path.splitext(filename)[0]
    out_path = os.path.join(IMG_ROOT, args.algo, subfolder, f"{name}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"✔ Saved visualization to {out_path}")


for fname, sfolder in files:
    run(fname, sfolder)

"""
Cách chạy:
    python visualizer_ortool.py --filename S042_N10_R_R3.json --subfolder S/N10
    python visualizer_ortool.py --filename S042_N10_R_R3.json
    python visualizer_ortool.py --subfolder S/N10
"""