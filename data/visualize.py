import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def visualize_tdtop_instance(file_path, output_image):
    # 1. Đọc dữ liệu instance
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Tạo thư mục đầu ra nếu chưa có
    output_image = Path(output_image)
    output_image.parent.mkdir(parents=True, exist_ok=True)

    # 3. Trích xuất thông tin
    general = data["Config"]["General"]
    vehicles = data["Config"]["Vehicles"]
    max_coord = general["MAX_COORD"]

    depot = data["Depot"]["coord"]
    road_nodes = data["RoadNodes"]
    info_nodes = data["InfoNodes"]

    road_coords = np.array([n["coord"] for n in road_nodes])
    info_coords = np.array([n["coord"] for n in info_nodes])
    info_p = np.array([n["p"] for n in info_nodes], dtype=float)

    # 4. Vẽ
    fig, ax = plt.subplots(figsize=(12, 10))

    # Nút thông tin (UAV phục vụ): màu + kích thước theo giá trị thông tin p_i
    if info_coords.size > 0:
        sizes = 20 + 80 * (info_p - info_p.min()) / max(np.ptp(info_p), 1e-9)
        sc = ax.scatter(
            info_coords[:, 0],
            info_coords[:, 1],
            c=info_p,
            cmap="viridis",
            s=sizes,
            marker="o",
            alpha=0.75,
            edgecolors="k",
            linewidths=0.3,
            label=f"Info node ({len(info_nodes)})",
            zorder=3,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Giá trị thông tin $p_i$")

    # Nút đường bộ (xe tải đi qua): hình thoi cam
    if road_coords.size > 0:
        ax.scatter(
            road_coords[:, 0],
            road_coords[:, 1],
            c="orange",
            marker="D",
            s=70,
            edgecolors="k",
            linewidths=0.8,
            label=f"Road node ({len(road_nodes)})",
            zorder=4,
        )
        # Ghi id nút đường bộ nếu số lượng không quá nhiều
        if len(road_nodes) <= 40:
            for n in road_nodes:
                ax.annotate(
                    str(n["id"]),
                    xy=n["coord"],
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color="darkred",
                    zorder=6,
                )

    # Kho (xuất phát và kết thúc cùng toạ độ)
    ax.scatter(
        depot[0],
        depot[1],
        c="red",
        marker="s",
        s=140,
        edgecolors="k",
        label="Depot",
        zorder=5,
    )

    # Khung vẽ theo MAX_COORD
    margin = 0.02 * max_coord
    ax.set_xlim(-margin, max_coord + margin)
    ax.set_ylim(-margin, max_coord + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")

    title = (
        f"Instance: {os.path.basename(file_path)}\n"
        f"Suite={general['SUITE']} | n={general['NUM_INFO_NODES']} | "
        f"dist={general['COORD_DISTRIBUTION']} | K={vehicles['NUM_TRUCKS']} | "
        f"R={vehicles['UAVS_PER_TRUCK'][0]} | "
        f"Battery={vehicles['BATTERY_MAH']} mAh | T_max={general['T_MAX_MIN']} min"
    )
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Lưu /img
    fig.tight_layout()
    fig.savefig(output_image, dpi=150)
    plt.close(fig)
    print(f"Visualization saved to: {output_image}")


def main():
    ap = argparse.ArgumentParser(
        description="Vẽ instance TD-TOP"
    )
    ap.add_argument("--filename", default=None,
                    help="tên file instance, vd: S042_N10_R_R3.json ")
    ap.add_argument("--subfolder", default="",
                    help="thư mục con bên trong data_dir, vd: S/N10. ")
    ap.add_argument("--data_dir", default="data", help="thư mục gốc chứa dữ liệu (mặc định: data)")
    ap.add_argument("--img_dir", default="img", help="thư mục gốc lưu ảnh (mặc định: img)")
    args = ap.parse_args()

    data_root = Path(args.data_dir)
    img_root = Path(args.img_dir)

    if not args.filename and not args.subfolder:
        ap.error("Cần ít nhất --filename hoặc --subfolder")

    if args.filename:
        name = args.filename[:-5] if args.filename.endswith(".json") else args.filename
        if args.subfolder:
            files = [data_root / args.subfolder / f"{name}.json"]
        else:
            files = sorted(data_root.rglob(f"{name}.json"))
            if not files:
                ap.error(f"Không tìm thấy {name}.json trong {data_root}")
            if len(files) > 1:
                print(f"Tìm thấy {len(files)} file trùng tên, vẽ tất cả:")
    else:
        in_dir = data_root / args.subfolder
        files = sorted(in_dir.glob("*.json"))
        if not files:
            ap.error(f"Không tìm thấy file .json nào trong {in_dir}")

    for fp in files:
        if not fp.exists():
            ap.error(f"Không tìm thấy file: {fp}")
        out_path = img_root / fp.parent.relative_to(data_root) / f"{fp.stem}.png"
        try:
            visualize_tdtop_instance(str(fp), out_path)
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Bỏ qua {fp}: không phải instance hợp lệ ({e!r})")


if __name__ == "__main__":
    main()