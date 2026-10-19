import argparse
import itertools
import math
import multiprocessing
import random
import time
from collections import deque
from pathlib import Path
from typing import Dict, List

from problem import Problem, Solution
from utils import (
    EPS,
    _Planner,
    _inner_stats,
    build_solution,
    decode,
    heuristic_individual,
    random_individual,
    cal_objective,
    solution_to_dict,
    save_result,
)


# =====================================================================
# Trạng thái lời giải của ALNS
# =====================================================================
class ALNSState(_Planner):
    """Hành trình xe + sortie theo cung; hỗ trợ xóa/chèn tăng dần."""

    # ---------------- khởi tạo / sao chép ----------------
    @classmethod
    def from_solution(cls, P: Problem, sol: Solution):
        st = cls(P, [list(plan.route.nodes) for plan in sol.plans])
        for plan in sol.plans:
            for s in plan.sorties:
                st.sorties[plan.truck][s.arc_index].append(
                    {"seq": list(s.nodes), "F": s.t_arrival - s.t_launch, "E": s.energy}
                )
        for k in range(P.num_trucks):
            for idx in range(len(st.nl[k]) - 1):
                st._refresh_arc(k, idx)
            st.cnt[k] = sum(len(lst) for lst in st.sorties[k])
        return st

    def copy(self):
        c = object.__new__(ALNSState)
        c.P = self.P
        c.nl = [list(x) for x in self.nl]
        c.tG = [list(x) for x in self.tG]
        c.dur = [list(x) for x in self.dur]
        c.sorties = [[list(lst) for lst in arcs] for arcs in self.sorties]
        c.used = list(self.used)
        c.cnt = list(self.cnt)
        c.violation = self.violation
        return c

    # ---------------- đại lượng dẫn xuất ----------------
    def _refresh_arc(self, k, idx):
        lst = self.sorties[k][idx]
        self.dur[k][idx] = max([self.tG[k][idx]] + [s["F"] for s in lst])
        self.used[k] = sum(self.dur[k])

    def viol(self):
        return sum(max(0.0, u - self.P.T) for u in self.used)

    def served_set(self):
        return {i for arcs in self.sorties for lst in arcs for so in lst for i in so["seq"]}

    def all_sorties(self):
        return [
            (k, idx, so)
            for k in range(len(self.nl))
            for idx in range(len(self.nl[k]) - 1)
            for so in self.sorties[k][idx]
        ]

    def road_nodes(self, k):
        return self.nl[k][1:-1]

    def signature(self):
        return hash(
            tuple(
                (
                    tuple(self.nl[k]),
                    tuple(
                        sorted(
                            (idx, tuple(s["seq"]))
                            for idx, lst in enumerate(self.sorties[k])
                            for s in lst
                        )
                    ),
                )
                for k in range(len(self.nl))
            )
        )

    def to_solution(self) -> Solution:
        self.violation = self.viol()
        return build_solution(self.P, self.nl, self)

    # ---------------- thay đổi hành trình xe ----------------
    def set_route(self, k, roads):
        """Đặt lại dãy nút đường của xe k. Sortie ở các cung còn nguyên (cùng cặp nút) được giữ,
        sortie ở cung biến mất được trả về dưới dạng các nhóm tự do."""
        P = self.P
        old_nl = self.nl[k]
        old_map = {
            (old_nl[i], old_nl[i + 1]): self.sorties[k][i] for i in range(len(old_nl) - 1)
        }
        nl = [0] + list(roads) + [P.end]
        new_pairs = set(zip(nl, nl[1:]))
        released = []
        for pair, lst in old_map.items():
            if pair not in new_pairs:
                released.extend(list(s["seq"]) for s in lst)
        tG = [P.truck_time[nl[i]][nl[i + 1]] for i in range(len(nl) - 1)]
        sorties = [list(old_map.get((nl[i], nl[i + 1]), [])) for i in range(len(nl) - 1)]
        self.nl[k], self.tG[k], self.sorties[k] = nl, tG, sorties
        self.dur[k] = [max([tG[i]] + [s["F"] for s in sorties[i]]) for i in range(len(tG))]
        self.used[k] = sum(self.dur[k])
        self.cnt[k] = sum(len(l) for l in sorties)
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

    # ---------------- xóa nút / sortie ----------------
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
            F, E = self._cost(new, _inner_stats(P, new), a, b, self.tG[k][idx])
            if E <= P.theta + EPS:
                lst[si] = {"seq": new, "F": F, "E": E}
            else:
                del lst[si]
                self.cnt[k] -= 1
                released.append(new)
        else:
            del lst[si]
            self.cnt[k] -= 1
        self._refresh_arc(k, idx)
        return released

    def remove_sortie_obj(self, k, idx, so):
        lst = self.sorties[k][idx]
        for j, x in enumerate(lst):
            if x is so:
                del lst[j]
                self.cnt[k] -= 1
                self._refresh_arc(k, idx)
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
            F, _ = self._cost(new, _inner_stats(P, new), a, b, self.tG[k][idx])
        return self.dur[k][idx] - max(self.tG[k][idx], F, other)

    # ---------------- chèn nút thông tin ----------------
    def arc_options(self, node, k, idx):
        """Các vị trí chèn khả thi của `node` trên cung (k,idx): tối đa 1 'sortie mới' và 1 'ghép tốt nhất'.
        Placement = (delta, E, k, idx, si (-1 nếu sortie mới), seq, F)."""
        P = self.P
        nl = self.nl[k]
        a, b = nl[idx], nl[idx + 1]
        # cận dưới năng lượng (bất đẳng thức tam giác): mọi đường bay a -> ... node ... -> b
        lb = P.uav_energy[a][node] + P.uav_energy[node][b] + P.p_srv * P.service[node]
        if lb > P.theta + EPS:
            return ()
        f_lb = P.uav_time[a][node] + P.uav_time[node][b] + P.service[node]
        if self.used[k] + max(0.0, f_lb - self.dur[k][idx]) > P.T + EPS:
            return ()
        tg = self.tG[k][idx]
        dur = self.dur[k][idx]
        lst = self.sorties[k][idx]
        res = []
        if len(lst) < P.R[k]:
            F, E = self._cost([node], (P.service[node], P.p_srv * P.service[node]), a, b, tg)
            if E <= P.theta + EPS:
                res.append((max(0.0, F - dur), E, k, idx, -1, [node], F))
        best_m = None
        for si, so in enumerate(lst):
            other = max([o["F"] for j, o in enumerate(lst) if j != si], default=0.0)
            seq = so["seq"]
            for pos in range(len(seq) + 1):
                new = seq[:pos] + [node] + seq[pos:]
                F, E = self._cost(new, _inner_stats(P, new), a, b, tg)
                if E > P.theta + EPS:
                    continue
                delta = max(tg, F, other) - dur
                if best_m is None or (delta, E) < best_m[:2]:
                    best_m = (delta, E, k, idx, si, new, F)
        if best_m is not None:
            res.append(best_m)
        return res

    def valid(self, pl):
        k = pl[2]
        if pl[4] == -1 and self.cnt[k] >= self.P.cap[k]:
            return False
        return self.used[k] + pl[0] <= self.P.T + EPS

    def apply_placement(self, pl):
        delta, E, k, idx, si, seq, F = pl
        lst = self.sorties[k][idx]
        if si == -1:
            lst.append({"seq": list(seq), "F": F, "E": E})
            self.cnt[k] += 1
        else:
            lst[si] = {"seq": list(seq), "F": F, "E": E}
        self._refresh_arc(k, idx)


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
    """Cùng công thức với cal_objective của GA."""
    served = st.served_set()
    f1 = sum(P.profit[i] for i in served)
    f2 = sum(s["E"] for _, _, s in st.all_sorties())
    V = st.viol()
    if V > EPS:
        fit = -(P.total_profit + 1.0) * (1.0 + V / P.T)
    else:
        fit = f1 - P.eps * f2
    return fit, f1, f2, V


def _pkey(pl):
    return (pl[0], pl[1])


# =====================================================================
# ALNS
# =====================================================================
class ALNSSolver:
    D_NAMES = [
        "D1_random", "D2_worst", "D3_shaw", "D4_sortie", "D5_window", "D6_wait",
        "D7_road_random", "D8_road_related", "D9_road_worst", "D10_road_segment",
    ]
    N_INFO_D = 6  # D1..D6 là toán tử trên nút thông tin
    RR_NAMES = ["R1_greedy", "R2_regret2"]
    RI_NAMES = ["I1_profit", "I2_cost", "I3_density", "I4_regret", "I5_noise"]

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
        w_start: float = 0.05,
        t_end_ratio: float = 0.01,
        sigma=(33.0, 9.0, 3.0),
        rho: float = 0.1,
        segment: int = 100,
        stag_raise: int = 200,
        reset_after: int = 500,
        reheat_mult: float = 5.0,
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

        self._max_ud = max(
            (self.P.uav_dist[i][j] for i in self.P.info_ids for j in self.P.info_ids), default=1.0
        ) or 1.0
        self._max_p = max((self.P.profit[i] for i in self.P.info_ids), default=1) or 1

    # ------------------------------------------------------------------
    # Toán tử phá hủy: trả về (nút đường bị gỡ, nhóm tự do)
    # ------------------------------------------------------------------
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
            idx = int(len(items) * (random.random() ** self.p_worst))
            out.append(items.pop(min(idx, len(items) - 1)))
        return out

    def _d_random(self, st, q, qR):
        served = list(st.served_set())
        nodes = random.sample(served, min(q, len(served)))
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
        seed_n = random.choice(served)
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
        k = random.choice(ks)
        n_arc = len(st.nl[k]) - 1
        start = random.randrange(n_arc)
        w = random.randint(1, 3)
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

    def _d_road_random(self, st, q, qR):
        roads = self._all_roads(st)
        nodes = random.sample(roads, min(qR, len(roads)))
        return nodes, st.remove_roads(nodes)

    def _d_road_related(self, st, q, qR):
        P = self.P
        roads = self._all_roads(st)
        seed_n = random.choice(roads)
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
        k = random.choice(ks)
        roads = st.road_nodes(k)
        w = min(qR, len(roads))
        start = random.randint(0, len(roads) - w)
        nodes = roads[start: start + w]
        return nodes, st.remove_roads(nodes)

    # ------------------------------------------------------------------
    # Toán tử sửa chữa
    # ------------------------------------------------------------------
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
            random.shuffle(pool)
            for x in pool:
                opt = min(self._road_options(st, x), key=lambda o: (o[0], o[1]))
                groups.extend(self._insert_road(st, x, opt))
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
            seq = list(g)
            while seq:
                plan = st.best_insertion(seq)
                if plan is not None:
                    st.apply_new(plan)
                    break
                seq.pop()  # nút bị cắt ở lại kho, sẽ được chèn lẻ ở bước sau

    def _repair_info(self, st, mode):
        P = self.P
        served = st.served_set()
        pool = [i for i in P.info_ids if P.reachable[i] and i not in served]
        if not pool:
            return
        cache = _Cache(st, pool)
        if mode == 0:  # I1: theo lợi nhuận giảm dần
            pool.sort(key=lambda n: (-P.profit[n], random.random()))
            for n in pool:
                opts = cache.options(n)
                cache.drop(n)
                if opts:
                    pl = min(opts, key=_pkey)
                    st.apply_placement(pl)
                    cache.refresh_arc(pl[2], pl[3])
            return

        k_reg = random.choice((2, 3))
        amp = self.noise * 0.1 * P.T
        while True:
            best_n, best_pl, best_sc = None, None, None
            for n in cache.nodes():
                opts = cache.options(n)
                if not opts:
                    continue
                pl = min(opts, key=_pkey)
                p = P.profit[n]
                if mode == 1:  # I2: chi phí nhỏ nhất
                    sc = (-pl[0], p, -pl[1])
                elif mode == 2:  # I3: mật độ lợi nhuận / chi phí
                    sc = (p / (pl[0] + 0.01), -pl[1])
                elif mode == 3:  # I4: regret-k trên các cung khác nhau
                    per_arc = {}
                    for o in opts:
                        a = (o[2], o[3])
                        c = o[0] + 0.05 * o[1] / P.theta
                        if a not in per_arc or c < per_arc[a]:
                            per_arc[a] = c
                    cs = sorted(per_arc.values())
                    reg = sum((cs[j] if j < len(cs) else P.T) - cs[0] for j in range(1, k_reg))
                    sc = ((reg + 0.01) * p, -pl[0])
                else:  # I5: chi phí có nhiễu
                    c = pl[0] + 0.05 * pl[1] / P.theta + random.uniform(0.0, amp)
                    sc = (-c, p)
                if best_sc is None or sc > best_sc:
                    best_n, best_pl, best_sc = n, pl, sc
            if best_n is None:
                break
            st.apply_placement(best_pl)
            cache.drop(best_n)
            cache.refresh_arc(best_pl[2], best_pl[3])

    # ------------------------------------------------------------------
    # Cải thiện cục bộ khi tìm được lời giải tốt nhất mới
    # ------------------------------------------------------------------
    def _polish(self, st):
        """Thử mọi hoán vị thứ tự bay của sortie (<= 6 nút) rồi lấp thêm nút còn lại."""
        P = self.P
        for k in range(len(st.nl)):
            for idx in range(len(st.nl[k]) - 1):
                a, b = st.nl[k][idx], st.nl[k][idx + 1]
                tg = st.tG[k][idx]
                lst = st.sorties[k][idx]
                changed = False
                for si, so in enumerate(lst):
                    seq = so["seq"]
                    if not (2 <= len(seq) <= 6):
                        continue
                    cur_key = (max(so["F"], tg), so["E"])
                    best = None
                    for perm in itertools.permutations(seq):
                        perm = list(perm)
                        F, E = st._cost(perm, _inner_stats(P, perm), a, b, tg)
                        if E > P.theta + EPS:
                            continue
                        key = (max(F, tg), E)
                        if key < cur_key and (best is None or key < best[0]):
                            best = (key, perm, F, E)
                    if best is not None:
                        lst[si] = {"seq": best[1], "F": best[2], "E": best[3]}
                        changed = True
                if changed:
                    st._refresh_arc(k, idx)
        self._repair_info(st, 1)

    # ------------------------------------------------------------------
    def _init_state(self):
        P = self.P
        best_sol, best_fit = None, None
        for t in range(self.init_tries):
            ind = heuristic_individual(P) if random.random() < 0.7 else random_individual(P)
            sol = decode(ind, P)
            fit = cal_objective(P, sol)
            if best_fit is None or fit[0] > best_fit[0]:
                best_sol, best_fit = sol, fit
        return ALNSState.from_solution(P, best_sol)

    @staticmethod
    def _roulette(weights, mask=None):
        w = [wi if (mask is None or mask[i]) else 0.0 for i, wi in enumerate(weights)]
        if sum(w) <= 0:
            return None
        return random.choices(range(len(w)), weights=w)[0]

    def run(self) -> Dict[str, Dict]:
        random.seed(self.seed)
        P = self.P
        start = time.time()
        history = {}

        d_fns = [
            self._d_random, self._d_worst, self._d_shaw, self._d_sortie, self._d_window, self._d_wait,
            self._d_road_random, self._d_road_related, self._d_road_worst, self._d_road_segment,
        ]
        nD, nRR, nRI = len(d_fns), len(self.RR_NAMES), len(self.RI_NAMES)
        w_d, w_rr, w_ri = [1.0] * nD, [1.0] * nRR, [1.0] * nRI
        seg_sc = {"d": [0.0] * nD, "rr": [0.0] * nRR, "ri": [0.0] * nRI}
        seg_n = {"d": [0] * nD, "rr": [0] * nRR, "ri": [0] * nRI}
        tot_n = {"d": [0] * nD, "rr": [0] * nRR, "ri": [0] * nRI}
        tot_best = {"d": [0] * nD, "rr": [0] * nRR, "ri": [0] * nRI}

        cur = self._init_state()
        cur_f, cur_f1, _, _ = state_objective(P, cur)
        cur_sig = cur.signature()
        best = cur.copy()
        best_f = cur_f
        best_obj = state_objective(P, best)

        T0 = -self.w_start * max(1.0, P.total_profit) / math.log(0.5)
        boost = 1.0
        stag = since_reset = resets_noimp = 0
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
            q = max(1, int(round(random.uniform(self.q_min, qmax_eff) * max(1, n_served))))
            qR = random.randint(1, max(1, int(round(qrmax_eff * nR)))) if nR else 0
            mask = [(i < self.N_INFO_D and n_served > 0) or (i >= self.N_INFO_D and nR > 0) for i in range(nD)]
            d = self._roulette(w_d, mask)
            if d is None:
                break

            new = cur.copy()
            road_pool, groups = d_fns[d](new, q, qR)
            rr = ri = None
            if road_pool:
                rr = self._roulette(w_rr)
                groups = list(groups) + self._repair_road(new, road_pool, rr)
            self._reinsert_groups(new, groups)
            ri = self._roulette(w_ri)
            self._repair_info(new, ri)

            fit_n, f1_n, f2_n, V_n = state_objective(P, new)
            sig = new.signature()
            score, accepted = 0.0, False
            improved_best = False
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
                    if random.random() < math.exp(min(0.0, fit_n - cur_f) / max(temp, 1e-12)):
                        score, accepted = sig2, True
            if accepted:
                cur, cur_f, cur_sig = new, fit_n, sig

            # ----- thống kê điểm toán tử -----
            for grp, j in (("d", d), ("rr", rr), ("ri", ri)):
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
                    cur = best.copy()
                    cur_f, cur_sig = best_f, cur.signature()
                    visited.clear()
                    visited_set.clear()
                    self.resets += 1
                    since_reset = 0
                    resets_noimp += 1
                    if resets_noimp >= 3:
                        boost = self.reheat_mult
                        resets_noimp = 0

            # ----- cập nhật trọng số cuối mỗi đoạn -----
            if it % self.segment == 0 and self.rho > 0:
                for grp, w in (("d", w_d), ("rr", w_rr), ("ri", w_ri)):
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


# =====================================================================
# Chạy nhiều bản độc lập song song (hạt giống khác nhau)
# =====================================================================
def _run_one(args):
    data_path, s_max, params, seed = args
    problem = Problem(data_path, s_max_default=s_max)
    solver = ALNSSolver(problem, seed=seed, **params)
    history = solver.run()
    solution = solver.best.to_solution()
    return history, solution, solver.op_stats, solver.resets, seed


def collect_files(data_root: Path, filename, subfolder):
    """Giống ortool.py / ga.py: trả về list[(đường dẫn file, thư mục con tương đối)]."""
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
        print("Cần truyền --filename và/hoặc --subfolder")
    return files


def main():
    """Run ALNS on datasets.

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
    parser.add_argument("--w-start", type=float, default=0.05)
    parser.add_argument("--t-end-ratio", type=float, default=0.01)
    parser.add_argument("--rho", type=float, default=0.1, help="Hệ số phản ứng; 0 = LNS thuần (trọng số cố định)")
    parser.add_argument("--segment", type=int, default=100)
    parser.add_argument("--stag-raise", type=int, default=200)
    parser.add_argument("--reset-after", type=int, default=500)
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
    result_dir = Path(__file__).parent / "result" / "alns"
    files = collect_files(data_root, args.filename, args.subfolder)
    if not files:
        return

    params = dict(
        max_iter=args.iterations, q_min=args.q_min, q_max=args.q_max, q_road_max=args.q_road_max,
        p_worst=args.p_worst, noise=args.noise, omega=args.omega, w_start=args.w_start,
        t_end_ratio=args.t_end_ratio, rho=args.rho, segment=args.segment,
        stag_raise=args.stag_raise, reset_after=args.reset_after, reheat_mult=args.reheat_mult,
        init_tries=args.init_tries, polish=not args.no_polish, time_limit=args.time_limit,
        verbose=(args.runs == 1), print_every=args.print_every,
    )

    print(f"\n{'=' * 80}\nALNS Solver (TD-TOP)\n{'=' * 80}\n")
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

    print(f"\n{'=' * 80}\nALNS experiments completed!\n{'=' * 80}\n")


if __name__ == "__main__":
    main()