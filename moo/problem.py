"""Bài toán TD-TOP đa xe tải đa mục tiêu (theo mô hình MILP milp/mo_hinh_TD-TOP_moi.md, milp/ortool.py).

    f1 = Σ p_i z_iq                          (36)  max
    f2 = Σ_q E_q,  E_q = fly + srv + hover    (37), (59)  min
Ràng buộc được giữ bởi bộ giải mã:
    * mọi nút đường được đúng một xe thăm (39)-(41); hành trình 0 -> ... -> 0'
    * mỗi nút thông tin phục vụ tối đa một lần (42); sortie cất cánh tại nút i và thu hồi tại điểm dừng kế tiếp j (48)
    * trên mỗi cung tối đa R_k sortie đồng thời (bảo toàn UAV (50)-(51)), mỗi xe tối đa |Q_k| = R_k * s_max_k sortie
      (dữ liệu không khai báo SORTIES_PER_TRUCK -> không giới hạn |Q_k|)
    * xe rời j khi mọi UAV đã về: tau_D_j = tau_D_i + max(t_ij^G, max_q F_q) (52)-(56); lơ lửng t_W = max(0, t^G - F) (57)
    * pin E_q <= theta (60); khung thời gian T (58) - nếu bị vi phạm thì ghi vào `violation`
Đơn vị nội bộ: thời gian giờ, năng lượng Wh, khoảng cách km.
"""

import json
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from config import EPS, KEY_HIGH, KEY_LOW, KNN_SIZE, S_MAX_DEFAULT

_PROBLEM_CACHE = {}


def load_problem(data_path, s_max_default: int = S_MAX_DEFAULT) -> "Problem":
    """Đọc (có cache theo tiến trình) - dùng khi unpickle Problem trong tiến trình con của multiprocessing."""
    key = (str(data_path), int(s_max_default))
    if key not in _PROBLEM_CACHE:
        Problem(data_path, s_max_default)
    return _PROBLEM_CACHE[key]


# ====================================================================
# Cấu trúc dữ liệu
# ====================================================================
@dataclass
class Node:
    id: int                   # chỉ số nội bộ: 0 = kho xuất phát, 1..num_road = nút đường, ... = nút thông tin, end = 0'
    orig_id: int              # id trong file dữ liệu
    coord: Tuple[float, float]
    kind: str                 # depot_start | road | info | depot_end
    profit: int = 0           # p_i
    service: float = 0.0      # s_i (giờ)


@dataclass
class Sortie:
    truck: int
    arc_index: int            # sortie bay trên cung thứ arc_index của lộ trình xe
    launch: int
    land: int
    nodes: List[int]
    t_launch: float
    t_arrival: float
    t_wait: float
    energy: float


@dataclass
class Route:
    nodes: List[int]
    arrival: List[float]
    departure: List[float]


@dataclass
class TruckPlan:
    truck: int
    route: Route
    sorties: List[Sortie]
    used: bool


class Solution:
    def __init__(self, plans: List[TruckPlan], violation: float):
        self.plans = plans
        self.violation = violation

    @property
    def served(self):
        return {i for p in self.plans for s in p.sorties for i in s.nodes}

    @property
    def energy(self):
        return sum(s.energy for p in self.plans for s in p.sorties)


# ====================================================================
# Bài toán
# ====================================================================
class Problem:
    n_obj = 2

    def __init__(self, data_path, s_max_default: int = S_MAX_DEFAULT):
        self.data_path = str(data_path)
        self.s_max_default = int(s_max_default)
        self._load(self.data_path)
        self._prepare()
        _PROBLEM_CACHE[(self.data_path, self.s_max_default)] = self

    def __reduce__(self):
        # chỉ gửi đường dẫn sang tiến trình con, ma trận được dựng lại (và cache) tại đó
        return load_problem, (self.data_path, self.s_max_default)

    def _load(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfgG, cfgV, cfgE = data["Config"]["General"], data["Config"]["Vehicles"], data["Config"]["Energy"]
        depot, road, info = data["Depot"], data["RoadNodes"], data["InfoNodes"]

        self.name = data_path
        self.T = float(cfgG["T_MAX_MIN"]) / 60.0
        self.num_trucks = int(cfgV["NUM_TRUCKS"])
        self.R = [int(r) for r in cfgV["UAVS_PER_TRUCK"]]
        self.v_truck = float(cfgV["V_TRUCK"])
        self.v_uav = float(cfgV["V_UAV"])
        self.theta = float(cfgV["THETA_WH"])
        self.p_srv = float(cfgE["P_SRV_W"])
        self.p_hover = float(cfgE["P_HOVER_W"])
        self.e_fly_per_km = float(cfgE["E_FLY_WH_PER_KM"])

        # |Q_k| = R_k * s_max_k; SORTIES_PER_TRUCK là tổng số sortie của một xe
        spt = cfgV.get("SORTIES_PER_TRUCK")
        unlimited = len(road) + 1
        self.s_max = [max(1, round(spt / self.R[k])) if spt and self.R[k] > 0
                      else (self.s_max_default if self.s_max_default > 0 else unlimited)
                      for k in range(self.num_trucks)]
        self.cap = [self.R[k] * self.s_max[k] for k in range(self.num_trucks)]

        self.nodes = [Node(0, depot["id_start"], tuple(depot["coord"]), "depot_start")]
        for nd in road:
            self.nodes.append(Node(len(self.nodes), nd["id"], tuple(nd["coord"]), "road"))
        for nd in info:
            self.nodes.append(Node(len(self.nodes), nd["id"], tuple(nd["coord"]), "info",
                                   int(nd["p"]), nd["s_min"] / 60.0))
        self.end = len(self.nodes)
        self.nodes.append(Node(self.end, depot["id_end"], tuple(depot["coord"]), "depot_end"))

        self.n_nodes = len(self.nodes)
        self.num_road = len(road)
        self.num_info = len(info)
        self.road_ids = list(range(1, 1 + self.num_road))
        self.info_ids = list(range(1 + self.num_road, 1 + self.num_road + self.num_info))
        self.orig_id = [n.orig_id for n in self.nodes]
        self.profit = [n.profit for n in self.nodes]
        self.service = [n.service for n in self.nodes]
        self.total_profit = sum(self.profit)

        coords = np.array([n.coord for n in self.nodes], dtype=float)
        diff = coords[:, None, :] - coords[None, :, :]
        manhattan = np.abs(diff).sum(-1)
        euclid = np.sqrt((diff ** 2).sum(-1))
        self.truck_dist_np = manhattan
        self.truck_dist = manhattan.tolist()
        self.uav_dist = euclid.tolist()
        self.truck_time = (manhattan / self.v_truck).tolist()      # t^G_ij
        self.uav_time = (euclid / self.v_uav).tolist()             # t^U_ij
        self.uav_energy = (euclid * self.e_fly_per_km).tolist()    # e^fly_ij = P^fly(v^U) t^U_ij

    def _prepare(self):
        launch_nodes = [0] + self.road_ids
        land_nodes = self.road_ids + [self.end]
        ut, ee = self.uav_time, self.uav_energy
        self.reachable = [False] * self.n_nodes
        self.e_single = [float("inf")] * self.n_nodes
        for i in self.info_ids:
            # cận dưới thời gian / năng lượng của sortie chỉ phục vụ i (chưa tính lơ lửng)
            f_min = min(ut[a][i] for a in launch_nodes) + self.service[i] + min(ut[i][b] for b in land_nodes)
            e_min = (min(ee[a][i] for a in launch_nodes) + self.p_srv * self.service[i]
                     + min(ee[i][b] for b in land_nodes))
            self.e_single[i] = e_min
            self.reachable[i] = e_min <= self.theta + EPS and f_min <= self.T + EPS
        self.reach_info = [i for i in self.info_ids if self.reachable[i]]
        # mức "đáng giá" của nút: giá trị / năng lượng tối thiểu
        self.score = [self.profit[i] / max(self.e_single[i], 1e-6) if self.reachable[i] else 0.0
                      for i in range(self.n_nodes)]
        # KNN_SIZE nút thông tin gần nhất (theo khoảng cách bay) của mỗi nút
        self.knn = {}
        for x in range(self.n_nodes):
            d = sorted(self.reach_info, key=lambda n: self.uav_dist[x][n])
            self.knn[x] = [n for n in d if n != x][:KNN_SIZE]

        # điểm chuẩn hóa: F1_REF = tổng giá trị nút với tới được; F2_REF = pin * số sortie tối đa có thể
        self.f1_ref = max(1.0, float(sum(self.profit[i] for i in self.reach_info)))
        self.f2_ref = max(1e-6, self.theta * min(sum(self.cap), max(1, len(self.reach_info))))

        self.n_var = self.num_road + self.num_info
        self.truck_width = (KEY_HIGH - KEY_LOW) / max(1, self.num_trucks)

    # ---------------------------------------------------------------- chuẩn hóa mục tiêu
    def normalize(self, f1, f2):
        return 1.0 - f1 / self.f1_ref, f2 / self.f2_ref

    def denormalize(self, o1, o2):
        return (1.0 - o1) * self.f1_ref, o2 * self.f2_ref

    # ---------------------------------------------------------------- random key <-> lời giải
    def road_keys(self, x):
        return x[:self.num_road]

    def info_keys(self, x):
        return x[self.num_road:]

    def truck_of_key(self, key):
        k = int((min(max(key, KEY_LOW), KEY_HIGH) - KEY_LOW) // self.truck_width)
        return min(max(k, 0), self.num_trucks - 1)

    def keys_to_routes(self, rkeys):
        """Khóa nút đường -> danh sách nút đường (theo thứ tự thăm) của từng xe."""
        routes = [[] for _ in range(self.num_trucks)]
        for j in np.argsort(np.asarray(rkeys, dtype=float), kind="stable"):
            routes[self.truck_of_key(rkeys[j])].append(self.road_ids[j])
        return routes

    def routes_to_keys(self, routes):
        """Ngược lại keys_to_routes: khóa đặt đều trong khoảng của xe, giữ thứ tự thăm."""
        keys = np.zeros(self.num_road)
        for k, r in enumerate(routes):
            lo = KEY_LOW + k * self.truck_width
            for pos, node in enumerate(r):
                keys[node - 1] = lo + self.truck_width * (pos + 1) / (len(r) + 1)
        return keys

    def selected_info(self, ikeys):
        """Nút thông tin được chọn (khóa > 0, với tới được), theo thứ tự khóa giảm dần = ưu tiên chèn."""
        ikeys = np.asarray(ikeys, dtype=float)
        order = np.argsort(-ikeys, kind="stable")
        return [self.info_ids[j] for j in order if ikeys[j] > 0 and self.reachable[self.info_ids[j]]]


def sortie_metrics(P: Problem, seq, a, b, tg):
    """Sortie a -> seq -> b trên cung xe có thời gian tg.
    Trả về (F, E, Ef, Es): F = bay + phục vụ, E = Ef + Es + P_hover * max(0, tg - F) (59)."""
    ut, ee, sv = P.uav_time, P.uav_energy, P.service
    first, last = seq[0], seq[-1]
    F = ut[a][first] + ut[last][b]
    Ef = ee[a][first] + ee[last][b]
    s = sv[first]
    prev = first
    for x in seq[1:]:
        F += ut[prev][x]
        Ef += ee[prev][x]
        s += sv[x]
        prev = x
    F += s
    Es = P.p_srv * s
    E = Ef + Es + P.p_hover * max(0.0, tg - F)
    return F, E, Ef, Es


# ====================================================================
# Trạng thái lời giải (dùng trong bộ giải mã và tìm kiếm cục bộ)
# ====================================================================
class State:
    """routes[k]: thứ tự nút đường của xe k;  sorties[k][idx]: các sortie trên cung idx của xe k,
    mỗi sortie là dict {seq, F, E, Ef, Es}.  dur[k][idx] = max(t^G, max F) - thời gian xe đi hết cung idx."""

    def __init__(self, P: Problem, routes):
        self.P = P
        self.routes = [list(r) for r in routes]
        self.nl = [[0] + r + [P.end] for r in self.routes]          # hành trình đầy đủ 0 -> ... -> 0'
        tt = P.truck_time
        self.tG = [[tt[nl[i]][nl[i + 1]] for i in range(len(nl) - 1)] for nl in self.nl]
        self.sorties = [[[] for _ in t] for t in self.tG]
        self.dur = [list(t) for t in self.tG]
        self.used = [sum(t) for t in self.tG]                        # thời điểm xe k về kho
        self.cnt = [0] * P.num_trucks                                # số sortie của xe k
        self.served = set()
        self.profit = 0
        self.energy = 0.0

    @property
    def violation(self):
        """Tổng thời gian vượt T (58), giờ."""
        T = self.P.T
        return sum(u - T for u in self.used if u > T)

    @property
    def feasible(self):
        return self.violation <= EPS

    @property
    def objectives(self):
        """(f1, f2) gốc."""
        return self.profit, self.energy

    def copy(self):
        n = State.__new__(State)
        n.P = self.P
        n.routes, n.nl, n.tG = self.routes, self.nl, self.tG          # không đổi sau khi tạo -> dùng chung
        n.sorties = [[[dict(s, seq=list(s["seq"])) for s in arc] for arc in tk] for tk in self.sorties]
        n.dur = [list(d) for d in self.dur]
        n.used = list(self.used)
        n.cnt = list(self.cnt)
        n.served = set(self.served)
        n.profit = self.profit
        n.energy = self.energy
        return n

    def refresh(self):
        """Tính lại dur, used, cnt, served, profit, energy từ sorties."""
        P = self.P
        served, E, prof = set(), 0.0, 0
        for k in range(P.num_trucks):
            c = 0
            for idx, lst in enumerate(self.sorties[k]):
                d = self.tG[k][idx]
                for s in lst:
                    c += 1
                    E += s["E"]
                    if s["F"] > d:
                        d = s["F"]
                    for n in s["seq"]:
                        served.add(n)
                        prof += P.profit[n]
                self.dur[k][idx] = d
            self.cnt[k] = c
            self.used[k] = sum(self.dur[k])
        self.served, self.energy, self.profit = served, E, prof

    def all_sorties(self):
        return [(k, idx, s) for k in range(self.P.num_trucks)
                for idx, lst in enumerate(self.sorties[k]) for s in lst]

    def has_free_slot(self):
        """Còn có thể thêm một sortie mới (còn |Q_k| và còn UAV rảnh trên ít nhất một cung)."""
        P = self.P
        for k in range(P.num_trucks):
            if self.cnt[k] < P.cap[k] and any(len(lst) < P.R[k] for lst in self.sorties[k]):
                return True
        return False

    def attach(self, k, idx, seq, F, E, Ef, Es):
        """Thêm sortie mới vào cung idx của xe k (đã kiểm tra khả thi bên ngoài)."""
        self.sorties[k][idx].append({"seq": list(seq), "F": F, "E": E, "Ef": Ef, "Es": Es})
        self.cnt[k] += 1
        if F > self.dur[k][idx]:
            self.used[k] += F - self.dur[k][idx]
            self.dur[k][idx] = F
        self.energy += E
        for n in seq:
            self.served.add(n)
            self.profit += self.P.profit[n]

    def detach(self, k, idx, s):
        """Gỡ sortie s khỏi cung idx của xe k (không cập nhật served/profit - người gọi tự xử lý)."""
        lst = self.sorties[k][idx]
        lst[:] = [o for o in lst if o is not s]
        self.cnt[k] -= 1
        d = max([self.tG[k][idx]] + [o["F"] for o in lst])
        self.used[k] += d - self.dur[k][idx]
        self.dur[k][idx] = d
        self.energy -= s["E"]

    def signature(self):
        return hash(tuple(
            (tuple(self.nl[k]), tuple(sorted((idx, tuple(s["seq"]))
                                            for idx, lst in enumerate(self.sorties[k]) for s in lst)))
            for k in range(self.P.num_trucks)))
