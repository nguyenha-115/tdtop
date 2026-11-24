"""Hằng số, đường dẫn và tham số mặc định cho các thuật toán đa mục tiêu trong moo/moo_algorithm.

Bài toán: TD-TOP đa xe tải (mô hình MILP trong milp/mo_hinh_TD-TOP_moi.md)
    f1 = tổng giá trị thông tin thu thập  (max)
    f2 = tổng năng lượng UAV, Wh          (min)
Các thuật toán trong moo_algorithm đều CỰC TIỂU HÓA, nên cal_fitness trả về vector chuẩn hóa
    o1 = 1 - f1 / F1_REF,   o2 = f2 / F2_REF      (lời giải khả thi luôn nằm trong [0, 1] x [0, 1])

Mã hóa (random key, mọi gen nằm trong [KEY_LOW, KEY_HIGH] = [-pi, pi] để khớp Individual.repair()):
    chromosome = [khóa nút đường (num_road)] + [khóa nút thông tin (num_info)]
    * khóa nút đường: chia [-pi, pi] thành K khoảng bằng nhau -> xe tải; trong một xe, khóa tăng dần = thứ tự thăm
    * khóa nút thông tin: > 0 -> nút được chọn phục vụ, khóa càng lớn càng được chèn sớm; <= 0 -> bỏ qua
"""

import argparse
import math
import sys
from pathlib import Path

# ----------------------------------------------------------------- đường dẫn
MOO_ROOT = Path(__file__).resolve().parent
ALGO_ROOT = MOO_ROOT / "moo_algorithm"
DATA_ROOT = MOO_ROOT.parent / "data" / "data"
RESULT_ROOT = MOO_ROOT / "result"

# moo_algorithm/*.py import "population", "metric" theo tên trần -> cần cả hai thư mục trong sys.path
for _p in (str(MOO_ROOT), str(ALGO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EPS = 1e-9
W_T = 200.0          # trọng số phạt vi phạm thời gian (giờ) khi so sánh trong bộ giải mã / tìm kiếm cục bộ
KNN_SIZE = 18        # số nút thông tin gần nhất của mỗi cung xe tải được xét khi chèn

# số sortie tối đa của mỗi UAV khi file dữ liệu không khai báo (0 = không giới hạn)
S_MAX_DEFAULT = 0
INFEASIBLE_OFFSET = 1.0  # lời giải vi phạm T có fitness >= 1 + INFEASIBLE_OFFSET ở cả hai mục tiêu

# ----------------------------------------------------------------- mã hóa random key
KEY_LOW, KEY_HIGH = -math.pi, math.pi
KEY_MIN_ACTIVE = 0.05    # khóa nút thông tin được bật sẽ nằm trong [KEY_MIN_ACTIVE, KEY_HIGH]

# ----------------------------------------------------------------- bộ giải mã
DECODER = {
    "route_2opt": True,   # 2-opt lộ trình xe tải sau khi phân xe theo khóa
    "opt_order": True,    # tối ưu thứ tự nút trong từng sortie
    "reposition": True,   # chuyển sortie sang cung khác nếu tiết kiệm năng lượng
    "second_pass": True,  # chèn lại các nút còn chỗ sau khi cải thiện
}

# ----------------------------------------------------------------- toán tử
INIT_FILL = (0.0, 1.0)   # tỉ lệ nút thông tin được bật trong cá thể khởi tạo ngẫu nhiên
INIT_GREEDY = 0.5        # tỉ lệ cá thể khởi tạo theo heuristic sweep + láng giềng gần nhất
SBX_ETA = 15.0
PM_ETA = 20.0
SBX_RATE = 0.3           # xác suất lai từng gen trong SBX
MAX_STEP = 3             # số bước đột biến tối đa của random_move

MUTATION_PROBS = {"add": 0.25, "drop": 0.25, "swap": 0.2, "route": 0.15, "poly": 0.15}
LS_RATE = 0.0            # xác suất local_search sau đột biến (ghi đè bằng --ls-rate)
LS_STEPS = 10

# ----------------------------------------------------------------- tham số mặc định từng thuật toán
ALGO_DEFAULT = {
    "nsga2": {"pop_size": 100, "max_gen": 100, "crossover_rate": 0.9, "mutation_rate": 0.3},
    "nsga3": {"pop_size": 100, "max_gen": 100, "crossover_rate": 0.9, "mutation_rate": 0.3},
    "moead": {"pop_size": 100, "max_gen": 100, "neighbor_size": 10},
    "mode": {"pop_size": 100, "max_gen": 100, "F": 0.5, "CR": 0.9},
    "mopso": {"pop_size": 100, "max_gen": 100, "w": 0.4, "c1": 1.5, "c2": 1.5},
    "pfgmoea": {"pop_size": 100, "max_gen": 100, "GK": 10, "sigma": 0.01,
                "crossover_rate": 0.9, "mutation_rate": 0.3},
}
for _v in ("rand1", "best1", "best1_pareto", "ctb1", "randbest1"):
    ALGO_DEFAULT[f"pfgmoea_{_v}"] = {"pop_size": 100, "max_gen": 200, "GK": 10, "sigma": 0.01, "F": 0.5, "CR": 0.9}

ALGO_NAMES = list(ALGO_DEFAULT)


def base_parser(description: str = "TD-TOP đa mục tiêu (moo_algorithm)") -> argparse.ArgumentParser:
    """Tham số dòng lệnh chung cho moo/utils.py và các script moo_algorithm/*.py."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--algo", type=str, default="nsga2", choices=ALGO_NAMES)
    p.add_argument("--filename", type=str, default=None, help="vd: S042_N10_R_R3.json")
    p.add_argument("--subfolder", type=str, default=None, help="vd: S/N10")
    p.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    p.add_argument("--processes", type=int, default=1, help="Số tiến trình tính fitness (1 = tuần tự, nhanh với instance nhỏ)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--s-max", type=int, default=S_MAX_DEFAULT,
                   help="Số sortie tối đa/UAV nếu dữ liệu không khai báo (0 = không giới hạn)")
    p.add_argument("--time-limit", type=float, default=None, help="Giới hạn thời gian mỗi instance (giây)")
    p.add_argument("--early-stop", type=int, default=None, help="Dừng nếu HV kho Pareto không tăng sau N thế hệ")
    # ghi đè tham số mặc định của thuật toán (None = dùng ALGO_DEFAULT)
    p.add_argument("--pop-size", type=int, default=None)
    p.add_argument("--max-gen", type=int, default=None)
    p.add_argument("--crossover-rate", type=float, default=None)
    p.add_argument("--mutation-rate", type=float, default=None)
    p.add_argument("--neighbor-size", type=int, default=None, help="MOEA/D: số bài toán con láng giềng")
    p.add_argument("--F", type=float, default=None, help="DE: hệ số co giãn")
    p.add_argument("--CR", type=float, default=None, help="DE: xác suất lai")
    p.add_argument("--w", type=float, default=None, help="MOPSO: quán tính")
    p.add_argument("--c1", type=float, default=None, help="MOPSO: hệ số cá nhân")
    p.add_argument("--c2", type=float, default=None, help="MOPSO: hệ số xã hội")
    p.add_argument("--GK", type=int, default=None, help="PFG-MOEA: số ô lưới")
    p.add_argument("--sigma", type=float, default=None, help="PFG-MOEA: biên lưới")
    p.add_argument("--ls-rate", type=float, default=LS_RATE, help="Xác suất tìm kiếm cục bộ sau đột biến")
    return p


def algo_params(args) -> dict:
    """Tham số của args.algo: mặc định trong ALGO_DEFAULT, ghi đè bởi các đối số dòng lệnh khác None."""
    params = dict(ALGO_DEFAULT[args.algo])
    for k in list(params):
        v = getattr(args, k, None)
        if v is None:
            continue
        params[k] = v
    return params
