import json
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class Node:
    id: int  # chỉ số nội bộ
    orig_id: int  # mã gốc trong file dữ liệu
    coord: Tuple[float, float]
    kind: str  # "depot_start" | "road" | "info" | "depot_end"
    profit: int = 0
    service: float = 0.0  # giờ


@dataclass
class Sortie:
    truck: int
    arc_index: int  # sortie nằm trên cung thứ mấy của hành trình xe
    launch: int
    land: int
    nodes: List[int]  # các nút thông tin theo thứ tự bay
    t_launch: float
    t_arrival: float  # thời điểm UAV về tới nút thu hồi (tA trong MILP)
    t_wait: float  # thời gian lơ lửng chờ xe
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
    def __init__(self, plans: List[TruckPlan], violation: float, signature: int):
        self.plans = plans
        self.violation = violation  # tổng thời gian xe vượt quá T (giờ)
        self.signature = signature  # mã băm kiểu hình, dùng để loại cá thể trùng

    @property
    def served(self):
        return {i for p in self.plans for s in p.sorties for i in s.nodes}

    @property
    def energy(self):
        return sum(s.energy for p in self.plans for s in p.sorties)


class Individual:
    """chromosome = [truck_layer, info_layer, break_mask]"""

    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = None  # (fitness, f1, f2, service_rate, violation, signature)


class Problem:
    def __init__(self, data_path, s_max_default: int = 2):
        self.s_max_default = s_max_default
        self.load_from_generated(data_path)

    def load_from_generated(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cfgG = data["Config"]["General"]
        cfgV = data["Config"]["Vehicles"]
        cfgE = data["Config"]["Energy"]
        depot = data["Depot"]
        road = data["RoadNodes"]
        info = data["InfoNodes"]

        # ---- tham số hệ thống (thời gian: giờ, năng lượng: Wh) ----
        self.T = float(cfgG["T_MAX_MIN"]) / 60.0
        self.num_trucks = int(cfgV["NUM_TRUCKS"])
        self.R = [int(r) for r in cfgV["UAVS_PER_TRUCK"]]
        self.v_truck = float(cfgV["V_TRUCK"])
        self.v_uav = float(cfgV["V_UAV"])
        self.theta = float(cfgV["THETA_WH"])
        self.p_srv = float(cfgE["P_SRV_W"])
        self.p_hover = float(cfgE["P_HOVER_W"])
        self.e_fly_per_km = float(cfgE["E_FLY_WH_PER_KM"])

        spt = cfgV.get("SORTIES_PER_TRUCK")
        self.s_max = []
        for k in range(self.num_trucks):
            if spt and self.R[k] > 0:
                self.s_max.append(max(1, round(spt / self.R[k])))
            else:
                self.s_max.append(self.s_max_default)
        # số sortie tối đa của từng xe: |Q_k| = R_k * s_max_k
        self.cap = [self.R[k] * self.s_max[k] for k in range(self.num_trucks)]

        # ---- nút ----
        self.nodes: List[Node] = []
        self.nodes.append(
            Node(0, depot["id_start"], tuple(depot["coord"]), "depot_start")
        )
        for nd in road:
            self.nodes.append(
                Node(len(self.nodes), nd["id"], tuple(nd["coord"]), "road")
            )
        for nd in info:
            self.nodes.append(
                Node(
                    len(self.nodes),
                    nd["id"],
                    tuple(nd["coord"]),
                    "info",
                    int(nd["p"]),
                    nd["s_min"] / 60.0,
                )
            )
        self.end = len(self.nodes)
        self.nodes.append(
            Node(self.end, depot["id_end"], tuple(depot["coord"]), "depot_end")
        )

        self.n_nodes = len(self.nodes)
        self.num_road = len(road)
        self.num_info = len(info)
        self.road_ids = list(range(1, 1 + self.num_road))
        self.info_ids = list(range(1 + self.num_road, 1 + self.num_road + self.num_info))
        # ký tự phân cách: K-1 giá trị >= delim_base
        self.delim_base = self.n_nodes
        self.delim_ids = list(range(self.delim_base, self.delim_base + self.num_trucks - 1))
        self.orig_id = [n.orig_id for n in self.nodes]

        self.profit = [n.profit for n in self.nodes]
        self.service = [n.service for n in self.nodes]
        self.total_profit = sum(self.profit)

        # ---- ma trận khoảng cách / thời gian / năng lượng ----
        coords = np.array([n.coord for n in self.nodes], dtype=float)
        diff = coords[:, None, :] - coords[None, :, :]
        manhattan = np.sum(np.abs(diff), axis=-1)
        euclid = np.sqrt(np.sum(diff**2, axis=-1))
        self.truck_dist = manhattan.tolist()
        self.uav_dist = euclid.tolist()
        self.truck_time = (manhattan / self.v_truck).tolist()  # t^G
        self.uav_time = (euclid / self.v_uav).tolist()  # t^U
        self.uav_energy = (euclid * self.e_fly_per_km).tolist()  # e^fly

        # ---- tiền xử lý nút thông tin ----
        truck_nodes = [0] + self.road_ids + [self.end]
        launch_nodes = [0] + self.road_ids
        land_nodes = self.road_ids + [self.end]
        self.reachable = [False] * self.n_nodes
        self.efficiency = [0.0] * self.n_nodes
        for i in self.info_ids:
            t_in = min(self.uav_time[a][i] for a in launch_nodes)
            t_out = min(self.uav_time[i][b] for b in land_nodes)
            e_in = min(self.uav_energy[a][i] for a in launch_nodes)
            e_out = min(self.uav_energy[i][b] for b in land_nodes)
            f_min = t_in + self.service[i] + t_out
            e_min = e_in + self.p_srv * self.service[i] + e_out
            # cận dưới: nếu đã vượt pin / khung thời gian thì không bao giờ phục vụ được
            self.reachable[i] = (e_min <= self.theta + 1e-9) and (f_min <= self.T + 1e-9)
            self.efficiency[i] = self.profit[i] / max(f_min, 1e-6)

        # hệ số ε để mục tiêu phụ (năng lượng) chỉ phân định khi f1 bằng nhau
        self.eps = 1.0 / (1.0 + sum(self.cap) * self.theta)