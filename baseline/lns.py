import argparse
import itertools
import math
import multiprocessing
import random
import time
from collections import deque
from pathlib import Path
from typing import Dict, List

import numpy as np

from problem import Problem, Solution
from utils import (
    EPS,
    State,
    prepare,
    metrics,
    opt_order,
    W_T,
    construct,
    random_routes,
    decode,
    cal_objective,
    verify_solution,
    solution_to_dict,
    save_result,
)


# Trạng thái lời giải của ALNS
class ALNSState(State):
    """Mở rộng utils.State (hành trình xe + sortie theo cung, mỗi sortie là dict
    {seq, F, E, Ef, Es}) bằng các thao tác xóa/chèn dùng cho ALNS.

    Quy ước: State.copy() dùng chung routes/nl/tG giữa các bản sao nên ở đây KHÔNG sửa tại chỗ
    các danh sách con của chúng; set_route() luôn thay cả phần tử bằng danh sách mới."""

    # khởi tạo / sao chép
    @classmethod
    def from_state(cls, st: State):
        return cls.copy(st)

    def copy(self):
        c = State.copy(self)
        c.__class__ = ALNSState
        c.routes = list(self.routes)
        c.nl = list(self.nl)
        c.tG = list(self.tG)
        return c

    # đại lượng dẫn xuất
    def _refresh_arc(self, k, idx):
        """Cập nhật nhanh thời gian cung / thời gian xe k (chưa đụng tới served, profit, energy)."""
        lst = self.sorties[k][idx]
        self.dur[k][idx] = max([self.tG[k][idx]] + [s["F"] for s in lst])
        self.used[k] = sum(self.dur[k])

    def viol(self):
        return sum(max(0.0, u - self.P.T) for u in self.used)

    def served_set(self):
        return set(self.served)

    def road_nodes(self, k):
        return self.nl[k][1:-1]

    def to_solution(self) -> Solution:
        self.refresh()
        return decode(self)

    # thay đổi hành trình xe
    def set_route(self, k, roads):
        """Đặt lại dãy nút đường của xe k. Sortie ở các cung còn nguyên (cùng cặp nút) được giữ,
        sortie ở cung biến mất được trả về dưới dạng các nhóm tự do."""
        P = self.P
        old_nl = self.nl[k]
        old_map = {(old_nl[i], old_nl[i + 1]): self.sorties[k][i] for i in range(len(old_nl) - 1)}
        nl = [0] + list(roads) + [P.end]
        new_pairs = set(zip(nl, nl[1:]))
        released = []
        for pair, lst in old_map.items():
            if pair not in new_pairs:
                released.extend(list(s["seq"]) for s in lst)
        tG = [P.truck_time[nl[i]][nl[i + 1]] for i in range(len(nl) - 1)]
        sorties = [list(old_map.get((nl[i], nl[i + 1]), [])) for i in range(len(nl) - 1)]
        self.routes[k] = list(roads)
        self.nl[k], self.tG[k], self.sorties[k] = nl, tG, sorties
        self.dur[k] = list(tG)
        self.refresh()
        return released

    def remove_roads(self, nodes):
        """Gỡ các nút đường; trả về các nhóm tự do bị giải phóng."""
        nodes = set(nodes)
        groups = []
        for k in range(len(self.nl)):
            cur = self.road_nodes(k)
            if any(x in nodes for x in cur):
                groups.extend(self.set_route(k, [x for x in cur if x not in nodes]))
        return groups

    # xóa nút / sortie
    def locate(self, node):
        for k in range(len(self.nl)):
            for idx in range(len(self.nl[k]) - 1):
                for si, so in enumerate(self.sorties[k][idx]):
                    if node in so["seq"]:
                        return k, idx, si
        return None

    def remove_node(self, node):
        """Gỡ nút khỏi sortie. Nếu sortie còn lại vượt pin (do thời gian lơ lửng tăng) thì giải phóng
        cả sortie. Trả về danh sách nhóm tự do (có thể rỗng)."""
        loc = self.locate(node)
        if loc is None:
            return []
        P = self.P
        k, idx, si = loc
        lst = self.sorties[k][idx]
        new = [x for x in lst[si]["seq"] if x != node]
        released = []
        if new:
            a, b = self.nl[k][idx], self.nl[k][idx + 1]
            F, E, Ef, Es = metrics(P, new, a, b, self.tG[k][idx])
            if E <= P.theta + EPS:
                lst[si] = {"seq": new, "F": F, "E": E, "Ef": Ef, "Es": Es, "t": True}
            else:
                del lst[si]
                released.append(new)
        else:
            del lst[si]
        self.refresh()
        return released

    def remove_sortie_obj(self, k, idx, so):
        lst = self.sorties[k][idx]
        for j, x in enumerate(lst):
            if x is so:
                del lst[j]
                self.refresh()
                return list(so["seq"])
        return None

    def removal_gain(self, node):
        """Thời gian cung tiết kiệm được nếu gỡ nút này."""
        loc = self.locate(node)
        if loc is None:
            return 0.0
        P = self.P
        k, idx, si = loc
        lst = self.sorties[k][idx]
        a, b = self.nl[k][idx], self.nl[k][idx + 1]
        other = max([o["F"] for j, o in enumerate(lst) if j != si], default=0.0)
        new = [x for x in lst[si]["seq"] if x != node]
        F = 0.0
        if new:
            F = metrics(P, new, a, b, self.tG[k][idx])[0]
        return self.dur[k][idx] - max(self.tG[k][idx], F, other)

    # chèn nút thông tin
    def _wt(self, k):
        """Giá quy đổi 1 giờ kéo dài hành trình ~ bao nhiêu Wh. Xe còn dư nhiều thời gian thì thời gian gần như
        miễn phí (ràng buộc thực sự là pin + số sortie); càng sát T càng đắt."""
        return W_T * min(1.0, max(0.02, (self.used[k] / self.P.T) ** 2))

    def arc_options(self, node, k, idx):
        """Các vị trí chèn khả thi của `node` trên cung (k,idx): 1 'sortie mới' + vị trí ghép tốt nhất cho TỪNG sortie.
        Placement = (cost, E, k, idx, si (-1 nếu sortie mới), seq, F, Ef, Es, delta, dE)
        với cost = dE + w_t * delta (đơn vị Wh); delta = thời gian cung tăng thêm; dE = năng lượng tăng thêm."""
        P = self.P
        ut, ee = P.uav_time, P.uav_energy
        nl = self.nl[k]
        a, b = nl[idx], nl[idx + 1]
        sn = P.service[node]
        pn = P.p_srv * sn
        # cận dưới năng lượng/thời gian (bất đẳng thức tam giác)
        lb = ee[a][node] + ee[node][b] + pn
        if lb > P.theta + EPS:
            return ()
        f_lb = ut[a][node] + ut[node][b] + sn
        if self.used[k] + max(0.0, f_lb - self.dur[k][idx]) > P.T + EPS:
            return ()
        tg = self.tG[k][idx]
        dur = self.dur[k][idx]
        lst = self.sorties[k][idx]
        th, ph = P.theta + EPS, P.p_hover
        wt = self._wt(k)
        res = []
        if len(lst) < P.R[k]:
            F, E, Ef, Es = metrics(P, [node], a, b, tg)
            if E <= th:
                delta = max(0.0, F - dur)
                res.append((E + wt * delta, E, k, idx, -1, [node], F, Ef, Es, delta, E))
        for si, so in enumerate(lst):
            other = max([o["F"] for j, o in enumerate(lst) if j != si], default=0.0)
            seq = so["seq"]
            F0, Ef0, Es0, E0 = so["F"], so["Ef"], so["Es"], so["E"]
            L = len(seq)
            prev = a
            best_m = None
            for pos in range(L + 1):
                nxt = seq[pos] if pos < L else b
                nF = F0 + ut[prev][node] + ut[node][nxt] - ut[prev][nxt] + sn
                nEf = Ef0 + ee[prev][node] + ee[node][nxt] - ee[prev][nxt]
                nEs = Es0 + pn
                hov = tg - nF
                nE = nEf + nEs + (ph * hov if hov > 0 else 0.0)
                if nE <= th:
                    delta = max(tg, nF, other) - dur
                    dE = nE - E0
                    cost = dE + wt * max(0.0, delta)
                    if best_m is None or cost < best_m[0]:
                        best_m = (cost, nE, k, idx, si, seq[:pos] + [node] + seq[pos:], nF, nEf, nEs, delta, dE)
                if pos < L:
                    prev = seq[pos]
            if best_m is not None:
                res.append(best_m)
        return res

    def valid(self, pl):
        k = pl[2]
        if pl[4] == -1 and self.cnt[k] >= self.P.cap[k]:
            return False
        return self.used[k] + pl[9] <= self.P.T + EPS

    def apply_placement(self, pl):
        cost, E, k, idx, si, seq, F, Ef, Es = pl[:9]
        if si == -1:
            self._attach(k, idx, seq, F, E, Ef, Es, touched=True)
        else:
            self.sorties[k][idx][si] = {"seq": list(seq), "F": F, "E": E, "Ef": Ef, "Es": Es, "t": True}
            self.refresh()


class _Cache:
    """Bộ nhớ đệm vị trí chèn theo (nút, cung). Sau mỗi lần chèn chỉ cần tính lại đúng cung bị thay đổi."""

    def __init__(self, st: ALNSState, nodes):
        self.st = st
        self.arcs = [(k, idx) for k in range(len(st.nl)) for idx in range(len(st.nl[k]) - 1)]
        self.opts = {}
        for n in nodes:
            d = {}
            for k, idx in self.arcs:
                o = st.arc_options(n, k, idx)
                if o:
                    d[(k, idx)] = o
            self.opts[n] = d

    def nodes(self):
        return list(self.opts.keys())

    def drop(self, n):
        self.opts.pop(n, None)

    def refresh_arc(self, k, idx):
        for n, d in self.opts.items():
            o = self.st.arc_options(n, k, idx)
            if o:
                d[(k, idx)] = o
            else:
                d.pop((k, idx), None)

    def options(self, n):
        st = self.st
        return [pl for lst in self.opts[n].values() for pl in lst if st.valid(pl)]


def state_objective(P: Problem, st: ALNSState):
    """Cùng công thức với cal_objective của GA (lấy trực tiếp từ State đã được duy trì đồng bộ)."""
    return st.score, st.profit, st.energy, st.violation


def _pkey(pl):
    return (pl[0], pl[1])

# ALNS
class ALNSSolver:
    D_NAMES = [
        "D1_random", "D2_worst", "D3_shaw", "D4_sortie", "D5_window", "D6_wait",
        "D7_road_random", "D8_road_related", "D9_road_worst", "D10_road_segment",
        "D11_target",
    ]
    N_INFO_D = 6  # D1..D6 là toán tử trên nút thông tin
    RR_NAMES = ["R1_greedy", "R2_regret2", "R3_potential"]
    RI_NAMES = ["I1_profit", "I2_cost", "I3_density", "I4_regret", "I5_noise", "I6_build"]
    M_NAMES = ["M1_plain", "M2_noise", "M3_swap"]  # chế độ sửa chữa (chọn thích nghi)

    def __init__(
        self,
        problem: Problem,
        max_iter: int = 3000,
        q_min: float = 0.10,
        q_max: float = 0.40,
        q_road_max: float = 0.30,
        p_worst: float = 3.0,
        noise: float = 0.1,
        omega: float = 1.0,
        w_start: float = 0.005,
        t_end_ratio: float = 0.01,
        sigma=(33.0, 9.0, 3.0),
        rho: float = 0.1,
        segment: int = 100,
        stag_raise: int = 200,
        reset_after: int = 300,
        reheat_mult: float = 5.0,
        elite_size: int = 8,
        q_kick: float = 0.5,
        init_tries: int = 20,
        polish: bool = True,
        time_limit: float = None,
        seed: int = 42,
        log_every: int = 10,
        verbose: bool = True,
        print_every: int = 100,
    ):
        self.P = problem
        self.max_iter = max_iter
        self.q_min, self.q_max, self.q_road_max = q_min, q_max, q_road_max
        self.p_worst, self.noise, self.omega = p_worst, noise, omega
        self.w_start, self.t_end_ratio = w_start, t_end_ratio
        self.sigma = sigma
        self.rho = rho
        self.segment = max(1, segment)
        self.stag_raise = max(1, stag_raise)
        self.reset_after = max(1, reset_after)
        self.reheat_mult = reheat_mult
        self.elite_size = max(1, elite_size)
        self.q_kick = q_kick
        self._prio, self._noisy = [], False
        self.init_tries = max(1, init_tries)
        self.polish_on = polish
        self.time_limit = time_limit
        self.seed = seed
        self.log_every = max(1, log_every)
        self.verbose = verbose
        self.print_every = print_every
        self.best = None
        self.op_stats = {}
        self.resets = 0
        self.rng = random.Random(seed)
        prepare(problem)  # láng giềng gần nhất cho các hàm dựng/chèn của utils

        self._max_ud = max(
            (self.P.uav_dist[i][j] for i in self.P.info_ids for j in self.P.info_ids), default=1.0
        ) or 1.0
        self._max_p = max((self.P.profit[i] for i in self.P.info_ids), default=1) or 1
        self._pot = self._arc_potential()

    def _arc_potential(self, top: int = 10):
        """pot[a][b]: tổng lợi nhuận của `top` nút thông tin có thể phục vụ bằng sortie đơn nút trên
        cung xe a->b (đã tính năng lượng lơ lửng chờ xe). Dùng để đánh giá vị trí chèn nút đường."""
        P = self.P
        tn = [0] + list(P.road_ids) + [P.end]
        info = list(P.info_ids)
        pot = {}
        if not info:
            return pot
        ue = np.array(P.uav_energy)
        ut = np.array(P.uav_time)
        tt = np.array(P.truck_time)
        prof = np.array([P.profit[i] for i in info], dtype=float)
        srv = np.array([P.service[i] for i in info])
        tn_a = np.array(tn)
        e_in, e_out = ue[np.ix_(tn_a, info)], ue[np.ix_(info, tn_a)]  # (A, I), (I, B)
        t_in, t_out = ut[np.ix_(tn_a, info)], ut[np.ix_(info, tn_a)]
        for ai, a in enumerate(tn):
            if a == P.end:
                continue
            F = t_in[ai][:, None] + srv[:, None] + t_out  # (I, B)
            hover = np.maximum(0.0, tt[a, tn_a][None, :] - F)
            E = e_in[ai][:, None] + P.p_srv * srv[:, None] + e_out + P.p_hover * hover
            ok = np.where(E <= P.theta + EPS, prof[:, None], 0.0)
            k = min(top, len(info))
            best = -np.partition(-ok, k - 1, axis=0)[:k].sum(axis=0)
            for bi, b in enumerate(tn):
                if b != a and b != 0:
                    pot[(a, b)] = float(best[bi])
        return pot

    # Toán tử phá hủy: trả về (nút đường bị gỡ, nhóm tự do)
    def _remove_nodes(self, st, nodes):
        groups = []
        for n in nodes:
            groups.extend(st.remove_node(n))
        gone = set(nodes)
        out = []
        for g in groups:
            g = [x for x in g if x not in gone]
            if g:
                out.append(g)
        return out

    def _randomized_pick(self, items, q):
        """Chọn q phần tử từ danh sách đã sắp (ưu tiên đầu danh sách) với độ ngẫu nhiên p_worst."""
        items = list(items)
        out = []
        while items and len(out) < q:
            idx = int(len(items) * (self.rng.random() ** self.p_worst))
            out.append(items.pop(min(idx, len(items) - 1)))
        return out

    def _d_random(self, st, q, qR):
        served = list(st.served_set())
        nodes = self.rng.sample(served, min(q, len(served)))
        return [], self._remove_nodes(st, nodes)

    def _d_worst(self, st, q, qR):
        served = list(st.served_set())
        P = self.P
        dens = {n: P.profit[n] / (st.removal_gain(n) + 0.01) for n in served}
        order = sorted(served, key=lambda n: dens[n])  # mật độ thấp nhất = tệ nhất
        return [], self._remove_nodes(st, self._randomized_pick(order, q))

    def _d_shaw(self, st, q, qR):
        P = self.P
        served = list(st.served_set())
        seed_n = self.rng.choice(served)
        loc = {}
        for k, idx, so in st.all_sorties():
            for i in so["seq"]:
                loc[i] = (id(so), (k, idx))

        def rel(j):
            r = P.uav_dist[seed_n][j] / self._max_ud
            r += 0.5 * abs(P.profit[seed_n] - P.profit[j]) / self._max_p
            if loc[seed_n][0] == loc[j][0]:
                r -= 0.3
            elif loc[seed_n][1] == loc[j][1]:
                r -= 0.15
            return r

        others = sorted((j for j in served if j != seed_n), key=rel)
        nodes = [seed_n] + others[: max(0, q - 1)]
        return [], self._remove_nodes(st, nodes)

    def _remove_sorties(self, st, items, q):
        removed_nodes, groups = 0, []
        for k, idx, so in items:
            if removed_nodes >= q:
                break
            g = st.remove_sortie_obj(k, idx, so)
            if g:
                groups.append(g)
                removed_nodes += len(g)
        return groups

    def _d_sortie(self, st, q, qR):
        P = self.P
        items = st.all_sorties()
        items.sort(key=lambda t: sum(P.profit[i] for i in t[2]["seq"]) / max(t[2]["E"], 1e-6))
        return [], self._remove_sorties(st, self._randomized_pick(items, len(items)), q)

    def _d_window(self, st, q, qR):
        ks = [k for k in range(len(st.nl)) if any(st.sorties[k][i] for i in range(len(st.nl[k]) - 1))]
        if not ks:
            return self._d_sortie(st, q, qR)
        k = self.rng.choice(ks)
        n_arc = len(st.nl[k]) - 1
        start = self.rng.randrange(n_arc)
        w = self.rng.randint(1, 3)
        items = [
            (k, idx, so)
            for idx in range(start, min(n_arc, start + w))
            for so in st.sorties[k][idx]
        ]
        if not items:
            return self._d_sortie(st, q, qR)
        return [], self._remove_sorties(st, items, 10**9)

    def _d_wait(self, st, q, qR):
        items = [(k, idx, so) for k, idx, so in st.all_sorties() if so["F"] > st.tG[k][idx] + EPS]
        if not items:
            return self._d_sortie(st, q, qR)
        items.sort(key=lambda t: -(t[2]["F"] - st.tG[t[0]][t[1]]))
        return [], self._remove_sorties(st, self._randomized_pick(items, len(items)), q)

    def _all_roads(self, st):
        return [x for k in range(len(st.nl)) for x in st.road_nodes(k)]

    def _d_target(self, st, q, qR):
        """Chọn một nút CHƯA phục vụ có lợi nhuận cao làm mục tiêu, gỡ các nút đang phục vụ gần nó
        (cùng cung / gần theo khoảng cách UAV) để giải phóng chỗ, rồi ép chèn mục tiêu trước."""
        P = self.P
        served = st.served_set()
        unserved = [i for i in P.info_ids if P.reachable[i] and i not in served]
        if not unserved or not served:
            return self._d_random(st, q, qR)
        u = self.rng.choices(unserved, weights=[P.profit[i] for i in unserved])[0]
        # cung "rẻ" nhất cho u theo cận dưới thời gian bay
        best_arc, best_lb = None, None
        for k in range(len(st.nl)):
            nl = st.nl[k]
            for idx in range(len(nl) - 1):
                a, b = nl[idx], nl[idx + 1]
                if P.uav_energy[a][u] + P.uav_energy[u][b] + P.p_srv * P.service[u] > P.theta + EPS:
                    continue
                lb = P.uav_time[a][u] + P.uav_time[u][b] + P.service[u] - st.dur[k][idx]
                if best_lb is None or lb < best_lb:
                    best_arc, best_lb = (k, idx), lb
        on_arc = set()
        if best_arc is not None:
            k, idx = best_arc
            on_arc = {i for so in st.sorties[k][idx] for i in so["seq"]}
        order = sorted(served, key=lambda j: P.uav_dist[u][j] - (1e9 if j in on_arc else 0.0))
        nodes = self._randomized_pick(order, max(1, q))
        # thêm vài nút chưa phục vụ khác quanh u làm mục tiêu phụ
        near = sorted((j for j in unserved if j != u), key=lambda j: P.uav_dist[u][j])
        self._prio = [u] + [j for j in near[:3] if self.rng.random() < 0.5]
        return [], self._remove_nodes(st, nodes)

    def _d_road_random(self, st, q, qR):
        roads = self._all_roads(st)
        nodes = self.rng.sample(roads, min(qR, len(roads)))
        return nodes, st.remove_roads(nodes)

    def _d_road_related(self, st, q, qR):
        P = self.P
        roads = self._all_roads(st)
        seed_n = self.rng.choice(roads)
        order = sorted(roads, key=lambda x: P.truck_dist[seed_n][x])
        nodes = order[: min(qR, len(order))]
        return nodes, st.remove_roads(nodes)

    def _d_road_worst(self, st, q, qR):
        P = self.P
        scored = []
        for k in range(len(st.nl)):
            nl = st.nl[k]
            for pos in range(1, len(nl) - 1):
                u, x, v = nl[pos - 1], nl[pos], nl[pos + 1]
                detour = P.truck_time[u][x] + P.truck_time[x][v] - P.truck_time[u][v]
                lost = sum(
                    P.profit[i]
                    for idx in (pos - 1, pos)
                    for so in st.sorties[k][idx]
                    for i in so["seq"]
                )
                scored.append((detour + self.omega * 0.01 * lost, x))
        scored.sort(key=lambda t: -t[0])
        nodes = self._randomized_pick([x for _, x in scored], qR)
        return nodes, st.remove_roads(nodes)

    def _d_road_segment(self, st, q, qR):
        ks = [k for k in range(len(st.nl)) if st.road_nodes(k)]
        k = self.rng.choice(ks)
        roads = st.road_nodes(k)
        w = min(qR, len(roads))
        start = self.rng.randint(0, len(roads) - w)
        nodes = roads[start: start + w]
        return nodes, st.remove_roads(nodes)

    # Toán tử sửa chữa
    def _road_options(self, st, x):
        """(vi phạm tăng thêm, chi phí, k, idx) cho mọi vị trí chèn nút đường x."""
        P = self.P
        res = []
        for k in range(len(st.nl)):
            nl = st.nl[k]
            over_now = max(0.0, st.used[k] - P.T)
            for idx in range(len(nl) - 1):
                a, b = nl[idx], nl[idx + 1]
                new_arc = P.truck_time[a][x] + P.truck_time[x][b]
                new_used = st.used[k] - st.dur[k][idx] + new_arc
                viol = max(0.0, new_used - P.T) - over_now
                lost = sum(P.profit[i] for so in st.sorties[k][idx] for i in so["seq"])
                cost = new_arc - st.dur[k][idx] + self.omega * 0.01 * lost
                res.append((viol, cost, k, idx))
        return res

    def _insert_road(self, st, x, opt):
        _, _, k, idx = opt
        roads = list(st.road_nodes(k))
        roads.insert(idx, x)
        return st.set_route(k, roads)

    def _repair_road(self, st, road_pool, mode):
        groups = []
        pool = list(road_pool)
        if mode == 0:  # tham lam
            self.rng.shuffle(pool)
            for x in pool:
                opt = min(self._road_options(st, x), key=lambda o: (o[0], o[1]))
                groups.extend(self._insert_road(st, x, opt))
        elif mode == 2:  # theo tiềm năng cung (lợi nhuận UAV phục vụ được), có nhiễu
            self.rng.shuffle(pool)
            pot = self._pot
            for x in pool:
                best_o, best_sc = None, None
                for o in self._road_options(st, x):
                    k, idx = o[2], o[3]
                    a, b = st.nl[k][idx], st.nl[k][idx + 1]
                    gain = pot.get((a, x), 0.0) + pot.get((x, b), 0.0) - pot.get((a, b), 0.0)
                    sc = (o[0], -gain * self.rng.uniform(0.7, 1.3) + 0.01 * o[1])
                    if best_sc is None or sc < best_sc:
                        best_o, best_sc = o, sc
                groups.extend(self._insert_road(st, x, best_o))
        else:  # regret-2 theo xe
            while pool:
                best_x, best_opt, best_reg = None, None, None
                for x in pool:
                    per_truck = {}
                    for o in self._road_options(st, x):
                        sc = o[0] * 1e3 + o[1]
                        if o[2] not in per_truck or sc < per_truck[o[2]][0]:
                            per_truck[o[2]] = (sc, o)
                    cands = sorted(per_truck.values(), key=lambda t: t[0])
                    reg = (cands[1][0] - cands[0][0]) if len(cands) > 1 else 0.0
                    if best_reg is None or reg > best_reg:
                        best_x, best_opt, best_reg = x, cands[0][1], reg
                groups.extend(self._insert_road(st, best_x, best_opt))
                pool.remove(best_x)
        return groups

    def _reinsert_groups(self, st, groups):
        P = self.P
        for g in sorted(groups, key=lambda g: -sum(P.profit[i] for i in g)):
            seq = [x for x in g if x not in st.served and P.reachable[x]]
            if seq:
                # đặt cả nhóm vào cung tốt nhất; nếu không được thì bỏ dần nút lợi nhuận thấp nhất
                # (nút bị bỏ sẽ được chèn lẻ ở bước _repair_info)
                st.place_best(seq)

    def _best_pl(self, opts):
        """Vị trí chèn tốt nhất; ở chế độ nhiễu, chi phí được cộng nhiễu ngẫu nhiên (Ropke & Pisinger)."""
        if not self._noisy:
            return min(opts, key=_pkey)
        amp = self.noise * 0.1 * self.P.theta
        return min(opts, key=lambda pl: (max(0.0, pl[0] + self.rng.uniform(-amp, amp)), pl[1]))

    def _repair_info(self, st, mode, prio=(), exclude=()):
        """Chèn nút chưa phục vụ. `prio`: chèn trước tiên (mục tiêu của D11);
        `exclude`: tạm hoãn (nút vừa bị gỡ, để nhường chỗ cho nút khác)."""
        P = self.P
        served = st.served_set()
        exclude = set(exclude)
        pool = [i for i in P.info_ids if P.reachable[i] and i not in served and i not in exclude]
        if not pool:
            return
        if mode == 5:  # I6: dựng sortie mới quanh nút hạt giống (utils.build_sortie) khi còn chỗ trống, rồi lấp tham lam
            for n in prio:
                opts = [pl for a_ in [(k_, i_) for k_ in range(len(st.nl)) for i_ in range(len(st.nl[k_]) - 1)]
                        for pl in st.arc_options(n, *a_) if st.valid(pl)]
                if opts:
                    st.apply_placement(min(opts, key=_pkey))
            guard = 0
            while st.has_free_slot() and guard < 6 and st.build_sortie(self.rng, 3, 0.2):
                guard += 1
            mode, prio = 2, ()
            pool = [i for i in P.info_ids if P.reachable[i] and i not in st.served and i not in exclude]
            if not pool:
                return
        cache = _Cache(st, pool)
        for n in prio:
            if n in cache.opts:
                opts = cache.options(n)
                cache.drop(n)
                if opts:
                    pl = min(opts, key=_pkey)
                    st.apply_placement(pl)
                    cache.refresh_arc(pl[2], pl[3])
        if mode == 0:  # I1: theo lợi nhuận giảm dần
            pool = cache.nodes()
            pool.sort(key=lambda n: (-P.profit[n], self.rng.random()))
            for n in pool:
                opts = cache.options(n)
                cache.drop(n)
                if opts:
                    pl = self._best_pl(opts)
                    st.apply_placement(pl)
                    cache.refresh_arc(pl[2], pl[3])
            return

        k_reg = self.rng.choice((2, 3))
        amp = self.noise * 0.1 * P.theta
        while True:
            best_n, best_pl, best_sc = None, None, None
            for n in cache.nodes():
                opts = cache.options(n)
                if not opts:
                    continue
                pl = self._best_pl(opts)
                p = P.profit[n]
                if mode == 1:  # I2: chi phí nhỏ nhất
                    sc = (-pl[0], p, -pl[1])
                elif mode == 2:  # I3: mật độ lợi nhuận / chi phí
                    sc = (p / (max(pl[0], 0.0) + 0.5), -pl[1])
                elif mode == 3:  # I4: regret-k trên các cung khác nhau
                    per_arc = {}
                    for o in opts:
                        a = (o[2], o[3])
                        c = o[0]
                        if a not in per_arc or c < per_arc[a]:
                            per_arc[a] = c
                    cs = sorted(per_arc.values())
                    reg = sum((cs[j] if j < len(cs) else P.theta) - cs[0] for j in range(1, k_reg))
                    sc = ((reg + 0.01) * p, -pl[0])
                else:  # I5: chi phí có nhiễu
                    c = pl[0] + self.rng.uniform(0.0, amp)
                    sc = (-c, p)
                if best_sc is None or sc > best_sc:
                    best_n, best_pl, best_sc = n, pl, sc
            if best_n is None:
                break
            st.apply_placement(best_pl)
            cache.drop(best_n)
            cache.refresh_arc(best_pl[2], best_pl[3])

    # Cải thiện cục bộ khi tìm được lời giải tốt nhất mới
    def _tidy(self, st):
        """Tối ưu thứ tự bay (2-opt + or-opt) cho các sortie vừa bị chạm; nếu giảm được năng lượng thì lấp thêm nút."""
        P = self.P
        gain = False
        for k, idx, so in st.all_sorties():
            if not so.pop("t", False) or len(so["seq"]) < 2:
                continue
            a, b, tg = st.nl[k][idx], st.nl[k][idx + 1], st.tG[k][idx]
            new = opt_order(P, so["seq"], a, b, tg)
            F, E, Ef, Es = metrics(P, new, a, b, tg)
            if E < so["E"] - 1e-7 and E <= P.theta + EPS:
                so.update(seq=new, F=F, E=E, Ef=Ef, Es=Es)
                gain = True
        if gain:
            st.refresh()
            self._repair_info(st, 2)

    def _exchange(self, st, max_moves=40):
        """Cục bộ hoán đổi: thay một nút đang phục vụ u bằng nút chưa phục vụ v (lợi nhuận cao hơn) ở vị trí rẻ nhất
        trong cùng sortie, rồi lấp thêm nút. Lặp tới khi không cải thiện được."""
        P = self.P
        ut, ee = P.uav_time, P.uav_energy
        th, ph, Tl = P.theta + EPS, P.p_hover, P.T + EPS
        for _ in range(max_moves):
            unserved = [n for n in P.info_ids if P.reachable[n] and n not in st.served]
            if not unserved:
                return
            best = None
            for k in range(len(st.nl)):
                for idx in range(len(st.nl[k]) - 1):
                    lst = st.sorties[k][idx]
                    a, b, tg, dur = st.nl[k][idx], st.nl[k][idx + 1], st.tG[k][idx], st.dur[k][idx]
                    for si, so in enumerate(lst):
                        other = max([o["F"] for j, o in enumerate(lst) if j != si], default=0.0)
                        seq = so["seq"]
                        for ui, u in enumerate(seq):
                            base = seq[:ui] + seq[ui + 1:]
                            pu = P.profit[u]
                            cands = [v for v in unserved if P.profit[v] >= pu]
                            if not cands:
                                continue
                            if base:
                                F0, _, Ef0, Es0 = metrics(P, base, a, b, tg)
                            for v in cands:
                                gain = P.profit[v] - pu
                                sv = P.service[v]
                                if not base:
                                    F, E, Ef, Es = metrics(P, [v], a, b, tg)
                                    opts = [(F, E, Ef, Es, [v])] if E <= th else []
                                else:
                                    opts = []
                                    L = len(base)
                                    prev = a
                                    for pos in range(L + 1):
                                        nxt = base[pos] if pos < L else b
                                        nF = F0 + ut[prev][v] + ut[v][nxt] - ut[prev][nxt] + sv
                                        nEf = Ef0 + ee[prev][v] + ee[v][nxt] - ee[prev][nxt]
                                        nEs = Es0 + P.p_srv * sv
                                        hov = tg - nF
                                        nE = nEf + nEs + (ph * hov if hov > 0 else 0.0)
                                        if nE <= th:
                                            opts.append((nF, nE, nEf, nEs, base[:pos] + [v] + base[pos:]))
                                        if pos < L:
                                            prev = base[pos]
                                for nF, nE, nEf, nEs, nseq in opts:
                                    delta = max(tg, nF, other) - dur
                                    if delta > EPS and st.used[k] + delta > Tl:
                                        continue
                                    key = (gain, -nE)
                                    if gain == 0 and nE >= so["E"] - 1e-7:
                                        continue
                                    if best is None or key > best[0]:
                                        best = (key, k, idx, si, nseq, nF, nE, nEf, nEs)
            if best is None:
                return
            _, k, idx, si, nseq, F, E, Ef, Es = best
            st.sorties[k][idx][si] = {"seq": nseq, "F": F, "E": E, "Ef": Ef, "Es": Es}
            st.refresh()
            self._repair_info(st, 2)

    def _polish(self, st):
        """Thử mọi hoán vị thứ tự bay của sortie (<= 6 nút) rồi lấp thêm nút còn lại."""
        P = self.P
        for k in range(len(st.nl)):
            for idx in range(len(st.nl[k]) - 1):
                a, b = st.nl[k][idx], st.nl[k][idx + 1]
                tg = st.tG[k][idx]
                lst = st.sorties[k][idx]
                for si, so in enumerate(lst):
                    seq = so["seq"]
                    if len(seq) < 2:
                        continue
                    cur_key = (max(so["F"], tg), so["E"])
                    best = None
                    perms = itertools.permutations(seq) if len(seq) <= 6 else [opt_order(P, seq, a, b, tg)]
                    for perm in perms:
                        perm = list(perm)
                        F, E, Ef, Es = metrics(P, perm, a, b, tg)
                        if E > P.theta + EPS:
                            continue
                        key = (max(F, tg), E)
                        if key < cur_key and (best is None or key < best[0]):
                            best = (key, perm, F, E, Ef, Es)
                    if best is not None:
                        lst[si] = {"seq": best[1], "F": best[2], "E": best[3], "Ef": best[4], "Es": best[5]}
        st.refresh()
        self._repair_info(st, 1)
        self._exchange(st)

    #
    def _init_state(self):
        """Khởi tạo: dựng nhiều lời giải bằng utils.construct (hành trình heuristic/ngẫu nhiên), lấy bản tốt nhất."""
        P = self.P
        best = None
        for t in range(self.init_tries):
            routes = random_routes(P, self.rng, heuristic=(t == 0 or self.rng.random() < 0.7))
            st = construct(P, routes, self.rng)
            if best is None or st.score > best.score:
                best = st
        st = ALNSState.from_state(best)
        if self.polish_on:
            self._polish(st)
        return st

    def _destroy_repair(self, cur, d_fn, q, qR, m, ri, w_rr):
        """Một bước phá hủy + sửa chữa trên bản sao của `cur`.
        m = 0: sửa chữa thường (chèn lại cả nhóm cũ); m = 1: có nhiễu, rã nhóm;
        m = 2: hoán đổi — nút vừa gỡ bị hoãn để nút chưa phục vụ được chèn trước."""
        new = cur.copy()
        before = new.served_set()
        self._prio, self._noisy = [], (m == 1)
        road_pool, groups = d_fn(new, q, qR)
        rr = None
        if road_pool:
            rr = self._roulette(w_rr)
            groups = list(groups) + self._repair_road(new, road_pool, rr)
        if m == 0:
            self._reinsert_groups(new, groups)
            self._repair_info(new, ri, self._prio)
        elif m == 1:
            self._repair_info(new, ri, self._prio)
        else:
            removed = before - new.served_set()
            self._repair_info(new, ri, self._prio, exclude=removed)
            self._repair_info(new, ri)
        self._noisy = False
        self._tidy(new)
        for _, _, so in new.all_sorties():
            so.pop("t", None)
        return new, rr

    def _update_elite(self, elite, f, sig, st):
        """Kho lời giải ưu tú, khác nhau về chữ ký, sắp giảm dần theo fitness."""
        if any(s == sig for _, s, _ in elite):
            return
        if len(elite) < self.elite_size or f > elite[-1][0]:
            elite.append((f, sig, st.copy()))
            elite.sort(key=lambda t: -t[0])
            del elite[self.elite_size:]

    def _pick_elite(self, elite):
        r = self.rng.choices(range(len(elite)), weights=[1.0 / (i + 1) for i in range(len(elite))])[0]
        return elite[r][2]

    def _kick(self, st):
        """Phá hủy mạnh (q_kick * số nút đang phục vụ, có thể đảo cả hành trình xe) + sửa chữa có nhiễu."""
        P = self.P
        n_served = len(st.served_set())
        q = max(1, int(round(self.rng.uniform(0.5, 1.0) * self.q_kick * max(1, n_served))))
        qR = self.rng.randint(1, max(1, P.num_road // 2)) if P.num_road and self.rng.random() < 0.5 else 0
        d_fn = self.rng.choice((self._d_random, self._d_shaw, self._d_target))
        new = st.copy()
        self._prio, self._noisy = [], True
        if qR:  # nhóm bị giải phóng trở thành nút chưa phục vụ, được chèn lại ở bước sửa chữa
            road_pool, _ = self._d_road_random(new, q, qR)
            self._repair_road(new, road_pool, 2)
        if new.served_set():
            d_fn(new, q, qR)
        self._repair_info(new, self.rng.randrange(len(self.RI_NAMES)), self._prio)
        self._noisy = False
        return new

    def _roulette(self, weights, mask=None):
        w = [wi if (mask is None or mask[i]) else 0.0 for i, wi in enumerate(weights)]
        if sum(w) <= 0:
            return None
        return self.rng.choices(range(len(w)), weights=w)[0]

    def run(self) -> Dict[str, Dict]:
        self.rng = random.Random(self.seed)
        P = self.P
        start = time.time()
        history = {}

        d_fns = [
            self._d_random, self._d_worst, self._d_shaw, self._d_sortie, self._d_window, self._d_wait,
            self._d_road_random, self._d_road_related, self._d_road_worst, self._d_road_segment,
            self._d_target,
        ]
        road_d = set(range(self.N_INFO_D, self.N_INFO_D + 4))
        nD, nRR, nRI, nM = len(d_fns), len(self.RR_NAMES), len(self.RI_NAMES), len(self.M_NAMES)
        w_d, w_rr, w_ri, w_m = [1.0] * nD, [1.0] * nRR, [1.0] * nRI, [1.0] * nM
        grps = (("d", w_d), ("rr", w_rr), ("ri", w_ri), ("m", w_m))
        seg_sc = {g: [0.0] * len(w) for g, w in grps}
        seg_n = {g: [0] * len(w) for g, w in grps}
        tot_n = {g: [0] * len(w) for g, w in grps}
        tot_best = {g: [0] * len(w) for g, w in grps}

        cur = self._init_state()
        cur_f, cur_f1, _, _ = state_objective(P, cur)
        cur_sig = cur.signature()
        best = cur.copy()
        best_f = cur_f
        best_obj = state_objective(P, best)
        elite = [(cur_f, cur_sig, cur.copy())]

        T0 = -self.w_start * max(1.0, P.total_profit) / math.log(0.5)
        boost = 1.0
        stag = since_reset = resets_noimp = 0
        self.n_dup = self.n_retry = 0
        visited = deque(maxlen=3000)
        visited_set = set()
        sig0, sig1, sig2 = self.sigma
        it = 0
        history[0] = self._record(best_obj, cur_f, T0, self.q_max, 0)

        while True:
            elapsed = time.time() - start
            progress = it / self.max_iter
            if self.time_limit:
                progress = max(progress, elapsed / self.time_limit)
            if progress >= 1.0:
                break
            temp = T0 * (self.t_end_ratio ** progress) * boost
            boost = 1.0 + (boost - 1.0) * 0.99
            it += 1

            served = cur.served_set()
            n_served = len(served)
            nR = P.num_road
            qmax_eff = min(0.8, self.q_max + 0.1 * (stag // self.stag_raise))
            qrmax_eff = min(0.8, self.q_road_max + 0.1 * (stag // self.stag_raise))
            q = max(1, int(round(self.rng.uniform(self.q_min, qmax_eff) * max(1, n_served))))
            qR = self.rng.randint(1, max(1, int(round(qrmax_eff * nR)))) if nR else 0
            mask = [(nR > 0) if i in road_d else (n_served > 0) for i in range(nD)]
            d = self._roulette(w_d, mask)
            if d is None:
                break
            m = self._roulette(w_m)
            ri = self._roulette(w_ri)
            new, rr = self._destroy_repair(cur, d_fns[d], q, qR, m, ri, w_rr)
            sig = new.signature()
            if sig == cur_sig and m != 2:
                # sửa chữa dựng lại y nguyên lời giải cũ: thử lại ngay ở chế độ hoán đổi
                self.n_retry += 1
                m = 2
                new, rr = self._destroy_repair(cur, d_fns[d], q, qR, m, ri, w_rr)
                sig = new.signature()

            fit_n, f1_n, f2_n, V_n = state_objective(P, new)
            score, accepted = 0.0, False
            improved_best = False
            if sig == cur_sig:
                self.n_dup += 1
            if sig != cur_sig and sig not in visited_set:
                if len(visited) == visited.maxlen:
                    visited_set.discard(visited[0])
                visited.append(sig)
                visited_set.add(sig)
                if fit_n > best_f + 1e-9:
                    if self.polish_on and V_n <= EPS:
                        self._polish(new)
                        fit_n, f1_n, f2_n, V_n = state_objective(P, new)
                        sig = new.signature()
                    best, best_f, best_obj = new.copy(), fit_n, (fit_n, f1_n, f2_n, V_n)
                    score, accepted, improved_best = sig0, True, True
                elif fit_n > cur_f + 1e-9:
                    score, accepted = sig1, True
                else:
                    if self.rng.random() < math.exp(min(0.0, fit_n - cur_f) / max(temp, 1e-12)):
                        score, accepted = sig2, True
            if accepted:
                cur, cur_f, cur_sig = new, fit_n, sig
                self._update_elite(elite, cur_f, cur_sig, cur)

            # ----- thống kê điểm toán tử -----
            for grp, j in (("d", d), ("rr", rr), ("ri", ri), ("m", m)):
                if j is None:
                    continue
                seg_n[grp][j] += 1
                seg_sc[grp][j] += score
                tot_n[grp][j] += 1
                if improved_best:
                    tot_best[grp][j] += 1

            # ----- đình trệ / quay về lời giải tốt nhất / tăng nhiệt độ -----
            if improved_best:
                stag = since_reset = resets_noimp = 0
            else:
                stag += 1
                since_reset += 1
                if since_reset >= self.reset_after:
                    # khởi động lại từ một lời giải ưu tú (không chỉ best) + cú hích phá hủy lớn
                    cur = self._kick(self._pick_elite(elite))
                    cur_f, cur_sig = state_objective(P, cur)[0], cur.signature()
                    self._update_elite(elite, cur_f, cur_sig, cur)
                    visited.clear()
                    visited_set.clear()
                    self.resets += 1
                    since_reset = 0
                    resets_noimp += 1
                    # tăng nhiệt độ dần theo số lần khởi động lại không cải thiện
                    boost = 1.0 + (self.reheat_mult - 1.0) * min(1.0, resets_noimp / 3.0)
                    if resets_noimp >= 3:
                        resets_noimp = 0

            # cập nhật trọng số cuối mỗi đoạn
            if it % self.segment == 0 and self.rho > 0:
                for grp, w in grps:
                    for j in range(len(w)):
                        if seg_n[grp][j] > 0:
                            w[j] = (1 - self.rho) * w[j] + self.rho * seg_sc[grp][j] / seg_n[grp][j]
                        w[j] = max(w[j], 0.05)
                    seg_sc[grp] = [0.0] * len(w)
                    seg_n[grp] = [0] * len(w)

            if it % self.log_every == 0:
                history[it] = self._record(best_obj, cur_f, temp, qmax_eff, self.resets)
            if self.verbose and self.print_every and it % self.print_every == 0:
                print(
                    "Iter {}: best={:.2f} f1={} cur={:.2f} T={:.3f} stag={} resets={}".format(
                        it, best_f, best_obj[1], cur_f, temp, stag, self.resets
                    )
                )

        history[it] = self._record(best_obj, cur_f, temp if it else T0, self.q_max, self.resets)
        self.best = best
        self.op_stats = {
            "destroy": {self.D_NAMES[j]: {"used": tot_n["d"][j], "new_best": tot_best["d"][j], "weight": w_d[j]} for j in range(nD)},
            "repair_road": {self.RR_NAMES[j]: {"used": tot_n["rr"][j], "new_best": tot_best["rr"][j], "weight": w_rr[j]} for j in range(nRR)},
            "repair_info": {self.RI_NAMES[j]: {"used": tot_n["ri"][j], "new_best": tot_best["ri"][j], "weight": w_ri[j]} for j in range(nRI)},
            "repair_mode": {self.M_NAMES[j]: {"used": tot_n["m"][j], "new_best": tot_best["m"][j], "weight": w_m[j]} for j in range(nM)},
            "duplicates": self.n_dup,
            "duplicate_retries": self.n_retry,
        }
        return history

    @staticmethod
    def _record(best_obj, cur_f, temp, qmax, resets):
        fit, f1, f2, V = best_obj
        return {
            "fitness": fit,
            "profit": f1,
            "energy": f2,
            "violation": V,
            "current_fitness": cur_f,
            "temperature": temp,
            "q_max": qmax,
            "resets": resets,
        }


# Chạy nhiều bản độc lập song song
def _run_one(args):
    data_path, s_max, params, seed = args
    problem = Problem(data_path, s_max_default=s_max)
    solver = ALNSSolver(problem, seed=seed, **params)
    history = solver.run()
    solution = solver.best.to_solution()
    return history, solution, solver.op_stats, solver.resets, seed


def collect_files(data_root: Path, filename, subfolder):
    files = []
    if filename and subfolder:
        path = data_root / subfolder / filename
        if not path.is_file():
            print(f"Không tìm thấy {path}")
            return []
        files.append((path, subfolder))
    elif filename:
        matches = sorted(data_root.glob(f"**/{filename}"))
        if not matches:
            print(f"Không tìm thấy {filename} trong {data_root}")
            return []
        for m in matches:
            files.append((m, m.parent.relative_to(data_root).as_posix()))
    elif subfolder:
        folder = data_root / subfolder
        if not folder.is_dir():
            print(f"Không tìm thấy thư mục {folder}")
            return []
        for f in sorted(folder.glob("*.json")):
            files.append((f, subfolder))
    else:
        print("--filename và/hoặc --subfolder")
    return files


def main():
    """
    Ví dụ:
        python alns.py --filename S042_N10_R_R3.json --subfolder S/N10
        python alns.py --filename S042_N10_R_R3.json
        python alns.py --subfolder S/N10
        python alns.py --subfolder L/N100 --runs 4 --processing-number 4
    """
    parser = argparse.ArgumentParser(description="Run ALNS solver for TD-TOP")
    parser.add_argument("--filename", type=str, default=None, help="Tên file JSON instance")
    parser.add_argument("--subfolder", type=str, default=None, help="Thư mục con dưới DATA_ROOT, vd S/N10 hoặc L/N100")
    parser.add_argument("--data-root", type=str, default="../data/data")
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--q-min", type=float, default=0.10)
    parser.add_argument("--q-max", type=float, default=0.40)
    parser.add_argument("--q-road-max", type=float, default=0.30)
    parser.add_argument("--p-worst", type=float, default=3.0)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--w-start", type=float, default=0.005)
    parser.add_argument("--t-end-ratio", type=float, default=0.01)
    parser.add_argument("--rho", type=float, default=0.1, help="Hệ số phản ứng; 0 = LNS thuần (trọng số cố định)")
    parser.add_argument("--segment", type=int, default=100)
    parser.add_argument("--stag-raise", type=int, default=200)
    parser.add_argument("--reset-after", type=int, default=300)
    parser.add_argument("--elite-size", type=int, default=8, help="Số lời giải ưu tú dùng để khởi động lại")
    parser.add_argument("--q-kick", type=float, default=0.5, help="Tỉ lệ phá hủy tối đa khi khởi động lại")
    parser.add_argument("--reheat-mult", type=float, default=5.0)
    parser.add_argument("--init-tries", type=int, default=20)
    parser.add_argument("--no-polish", action="store_true")
    parser.add_argument("--time-limit", type=float, default=None, help="Giới hạn thời gian mỗi instance (giây)")
    parser.add_argument("--s-max", type=int, default=2)
    parser.add_argument("--runs", type=int, default=1, help="Số bản chạy độc lập (hạt giống seed, seed+1, ...)")
    parser.add_argument("--processing-number", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=100, help="In tiến trình mỗi N vòng lặp (khi --runs 1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    result_dir = Path(__file__).parent / "result" / "lns"
    files = collect_files(data_root, args.filename, args.subfolder)
    if not files:
        return

    params = dict(
        max_iter=args.iterations, q_min=args.q_min, q_max=args.q_max, q_road_max=args.q_road_max,
        p_worst=args.p_worst, noise=args.noise, omega=args.omega, w_start=args.w_start,
        t_end_ratio=args.t_end_ratio, rho=args.rho, segment=args.segment,
        stag_raise=args.stag_raise, reset_after=args.reset_after, reheat_mult=args.reheat_mult,
        elite_size=args.elite_size, q_kick=args.q_kick,
        init_tries=args.init_tries, polish=not args.no_polish, time_limit=args.time_limit,
        verbose=(args.runs == 1), print_every=args.print_every,
    )

    print(f"\n{'=' * 80}\nLNS Solver (TD-TOP)\n{'=' * 80}\n")
    print(f"Found {len(files)} file(s)")

    for data_file, sub in files:
        print(f"\n{'-' * 80}\nProcessing: {sub}/{data_file.name}\n{'-' * 80}")
        try:
            problem = Problem(str(data_file), s_max_default=args.s_max)
            start_time = time.time()
            jobs = [(str(data_file), args.s_max, params, args.seed + r) for r in range(args.runs)]
            if args.runs > 1 and args.processing_number > 1:
                with multiprocessing.Pool(min(args.processing_number, args.runs)) as pool:
                    outs = pool.map(_run_one, jobs)
            else:
                outs = [_run_one(j) for j in jobs]
            end_time = time.time()

            scored = [(cal_objective(problem, o[1]), o) for o in outs]
            for fit_i, o in scored:
                errs = verify_solution(problem, o[1])
                if errs:
                    print(f"  [CẢNH BÁO] seed {o[4]} vi phạm ràng buộc: {errs[:3]}")
            fit, (history, solution, op_stats, resets, seed_used) = max(scored, key=lambda t: t[0][0])

            result = {
                "time": end_time - start_time,
                "history": history,
                "best_fitness": fit[0],
                "best_profit": fit[1],
                "best_energy": fit[2],
                "best_service_rate": len(solution.served) / max(1, problem.num_info),
                "num_resets": resets,
                "best_seed": seed_used,
                "runs_best_profit": [t[0][1] for t in scored],
                "operator_stats": op_stats,
                "best_solution": solution_to_dict(problem, solution, fit),
            }
            save_result(result, result_dir / sub / data_file.name)
            print(f"  Completed in {result['time']:.2f} seconds")
            print(f"  Best profit f1: {result['best_profit']:.2f}  (runs: {result['runs_best_profit']})")

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\n{'=' * 80}\nLNS experiments completed!\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
