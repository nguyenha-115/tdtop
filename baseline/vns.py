import argparse
import itertools
import math
import multiprocessing
import random
import time
from collections import deque
from pathlib import Path
from typing import Dict, Any

import numpy as np

from problem import Problem, Solution
from utils import (
    EPS,
    _Planner,
    _inner_stats,
    build_solution,
    decode,
    heuristic_individual,
    random_individual,
    solution_to_dict,
    save_result,
)


def f1_objective(problem: Problem, solution: Solution):
    f1 = sum(problem.profit[i] for i in solution.served)
    f2 = solution.energy
    if solution.violation > EPS:
        fitness = -(problem.total_profit + 1.0) * (1.0 + solution.violation / problem.T)
    else:
        fitness = float(f1)
    return fitness, f1, f2


class VNSState(_Planner):
    @classmethod
    def from_solution(cls, P: Problem, sol: Solution):
        st = cls(P, [list(plan.route.nodes) for plan in sol.plans])
        for plan in sol.plans:
            for s in plan.sorties:
                st.sorties[plan.truck][s.arc_index].append(
                    {"seq": list(s.nodes), "F": s.t_arrival - s.t_launch, "E": s.energy}
                )
        st.recompute_all()
        return st

    def copy(self):
        c = object.__new__(VNSState)
        c.P = self.P
        c.nl = [list(x) for x in self.nl]
        c.tG = [list(x) for x in self.tG]
        c.dur = [list(x) for x in self.dur]
        c.sorties = [[list(lst) for lst in arcs] for arcs in self.sorties]
        c.used = list(self.used)
        c.cnt = list(self.cnt)
        c.violation = self.violation
        return c

    def assign(self, other: "VNSState"):
        """Nhận toàn bộ nội dung của `other` (other là bản sao dùng một lần)."""
        self.nl, self.tG, self.dur = other.nl, other.tG, other.dur
        self.sorties, self.used, self.cnt = other.sorties, other.used, other.cnt
        self.violation = other.violation

    # ---------- đại lượng dẫn xuất ----------
    def refresh_arc(self, k, idx):
        d = self.tG[k][idx]
        for so in self.sorties[k][idx]:
            if so["F"] > d:
                d = so["F"]
        self.dur[k][idx] = d
        self.used[k] = sum(self.dur[k])

    def recompute_all(self):
        """Tính lại chính xác F, E của mọi sortie (loại bỏ sai số cộng dồn)."""
        P = self.P
        for k in range(len(self.nl)):
            nl = self.nl[k]
            for idx in range(len(nl) - 1):
                a, b, tg = nl[idx], nl[idx + 1], self.tG[k][idx]
                lst = self.sorties[k][idx]
                for si, so in enumerate(lst):
                    F, E = self._cost(so["seq"], _inner_stats(P, so["seq"]), a, b, tg)
                    lst[si] = {"seq": list(so["seq"]), "F": F, "E": E}
            self.dur[k] = [
                max([self.tG[k][i]] + [s["F"] for s in self.sorties[k][i]]) for i in range(len(nl) - 1)
            ]
            self.used[k] = sum(self.dur[k])
            self.cnt[k] = sum(len(l) for l in self.sorties[k])
        self.violation = self.viol()

    def viol(self):
        return sum(max(0.0, u - self.P.T) for u in self.used)

    def arcs(self, trucks=None):
        ks = range(len(self.nl)) if trucks is None else trucks
        return [(k, idx) for k in ks for idx in range(len(self.nl[k]) - 1)]

    def served_set(self):
        return {i for arcs in self.sorties for lst in arcs for so in lst for i in so["seq"]}

    def locations(self):
        loc = {}
        for k in range(len(self.nl)):
            for idx, lst in enumerate(self.sorties[k]):
                for si, so in enumerate(lst):
                    for i in so["seq"]:
                        loc[i] = (k, idx, si)
        return loc

    def locate(self, node):
        for k in range(len(self.nl)):
            for idx, lst in enumerate(self.sorties[k]):
                for si, so in enumerate(lst):
                    if node in so["seq"]:
                        return k, idx, si
        return None

    def objective(self):
        """(fitness, f1, f2, violation). fitness chỉ phụ thuộc f1 (và mức vi phạm T); f2 chỉ để ghi nhận."""
        P = self.P
        f1, f2 = 0, 0.0
        for arcs in self.sorties:
            for lst in arcs:
                for so in lst:
                    f2 += so["E"]
                    for i in so["seq"]:
                        f1 += P.profit[i]
        V = self.viol()
        if V > EPS:
            fit = -(P.total_profit + 1.0) * (1.0 + V / P.T)
        else:
            fit = float(f1)
        return fit, f1, f2, V

    def signature(self):
        return hash(
            tuple(
                (
                    tuple(self.nl[k]),
                    tuple(sorted((idx, tuple(s["seq"])) for idx, lst in enumerate(self.sorties[k]) for s in lst)),
                )
                for k in range(len(self.nl))
            )
        )

    def to_solution(self) -> Solution:
        self.recompute_all()
        return build_solution(self.P, self.nl, self)

    # sao lưu / khôi phục một cung
    def snapshot(self, k, idx):
        return (k, idx, list(self.sorties[k][idx]), self.dur[k][idx], self.used[k], self.cnt[k])

    def restore(self, snap):
        k, idx, lst, d, u, c = snap
        self.sorties[k][idx] = lst
        self.dur[k][idx] = d
        self.used[k] = u
        self.cnt[k] = c

    # hành trình xe tải
    def set_route(self, k, roads):
        """Đặt lại dãy nút đường của xe k. Sortie trên cung còn giữ (cùng cặp nút, hoặc cặp đảo chiều
        với đường bay đảo ngược — khoảng cách đối xứng nên F, E không đổi) được giữ nguyên;
        sortie trên cung biến mất được trả về dưới dạng các nhóm tự do."""
        P = self.P
        old = self.nl[k]
        old_map = {(old[i], old[i + 1]): self.sorties[k][i] for i in range(len(old) - 1)}
        nl = [0] + list(roads) + [P.end]
        taken = set()
        sorties = []
        for i in range(len(nl) - 1):
            a, b = nl[i], nl[i + 1]
            if (a, b) in old_map:
                sorties.append(list(old_map[(a, b)]))
                taken.add((a, b))
            elif (b, a) in old_map:
                sorties.append([{"seq": so["seq"][::-1], "F": so["F"], "E": so["E"]} for so in old_map[(b, a)]])
                taken.add((b, a))
            else:
                sorties.append([])
        released = [list(s["seq"]) for pair, lst in old_map.items() if pair not in taken for s in lst]
        self.nl[k] = nl
        self.tG[k] = [P.truck_time[nl[i]][nl[i + 1]] for i in range(len(nl) - 1)]
        self.sorties[k] = sorties
        self.dur[k] = [max([self.tG[k][i]] + [s["F"] for s in sorties[i]]) for i in range(len(nl) - 1)]
        self.used[k] = sum(self.dur[k])
        self.cnt[k] = sum(len(l) for l in sorties)
        return released

    # ---------- xóa ----------
    def remove_sortie(self, k, idx, si):
        so = self.sorties[k][idx].pop(si)
        self.cnt[k] -= 1
        self.refresh_arc(k, idx)
        return list(so["seq"])

    def remove_node(self, node, loc=None):
        """Gỡ nút khỏi sortie (cập nhật F, E trong O(1)). Nếu phần còn lại vượt pin (thời gian lơ lửng
        tăng) thì giải phóng cả sortie. Trả về danh sách các nhóm bị giải phóng (thường rỗng)."""
        loc = loc or self.locate(node)
        if loc is None:
            return []
        P = self.P
        k, idx, si = loc
        lst = self.sorties[k][idx]
        so = lst[si]
        seq = so["seq"]
        if len(seq) == 1:
            self.remove_sortie(k, idx, si)
            return []
        a, b, tg = self.nl[k][idx], self.nl[k][idx + 1], self.tG[k][idx]
        pos = seq.index(node)
        prev = a if pos == 0 else seq[pos - 1]
        nxt = b if pos == len(seq) - 1 else seq[pos + 1]
        tU, eU, s = P.uav_time, P.uav_energy, P.service[node]
        F = so["F"] + tU[prev][nxt] - tU[prev][node] - tU[node][nxt] - s
        Ea = so["E"] - P.p_hover * max(0.0, tg - so["F"])
        Ea += eU[prev][nxt] - eU[prev][node] - eU[node][nxt] - P.p_srv * s
        E = Ea + P.p_hover * max(0.0, tg - F)
        new = seq[:pos] + seq[pos + 1:]
        if E > P.theta + EPS:
            self.remove_sortie(k, idx, si)
            return [new]
        lst[si] = {"seq": new, "F": F, "E": E}
        self.refresh_arc(k, idx)
        return []

    # ---------- chèn ----------
    def arc_options(self, n, k, idx, w):
        """Vị trí chèn khả thi của nút `n` trên cung (k, idx).
        Placement = (delta_time, dE, cost, k, idx, si (-1: sortie mới), pos, F, E, n).
        Trả về tối đa 3 phương án: sortie mới, ghép rẻ nhất theo cost, ghép có delta nhỏ nhất."""
        P = self.P
        nl = self.nl[k]
        a, b = nl[idx], nl[idx + 1]
        tU, eU = P.uav_time, P.uav_energy
        s = P.service[n]
        th, T, ph = P.theta, P.T, P.p_hover
        lb_e = eU[a][n] + eU[n][b] + P.p_srv * s  # cận dưới (bất đẳng thức tam giác)
        if lb_e > th + EPS:
            return ()
        dur = self.dur[k][idx]
        lb_t = tU[a][n] + tU[n][b] + s
        if self.used[k] + max(0.0, lb_t - dur) > T + EPS:
            return ()
        tg = self.tG[k][idx]
        lst = self.sorties[k][idx]
        res = []
        if len(lst) < P.R[k]:
            E = lb_e + ph * max(0.0, tg - lb_t)
            if E <= th + EPS:
                d = max(0.0, lb_t - dur)
                res.append((d, E, d / T + w * E / th, k, idx, -1, 0, lb_t, E, n))
        if lst:
            # F lớn nhất và lớn thứ hai trên cung -> "max F của các sortie khác" trong O(1)
            i1, f1, f2 = -1, 0.0, 0.0
            for j, so in enumerate(lst):
                if so["F"] > f1:
                    i1, f1, f2 = j, so["F"], f1
                elif so["F"] > f2:
                    f2 = so["F"]
            best_c = best_d = None
            bc_cost = bd_d = bd_e = float("inf")
            tUn, eUn = tU[n], eU[n]  # ma trận đối xứng: tU[x][n] == tU[n][x]
            es = P.p_srv * s
            lim = th + EPS
            inv_T, w_th = 1.0 / T, w / th
            for si, so in enumerate(lst):
                other = f2 if si == i1 else f1
                base = tg if tg > other else other
                seq = so["seq"]
                F0, E0 = so["F"], so["E"]
                Ea0 = E0 - (ph * (tg - F0) if tg > F0 else 0.0)
                prev = a
                L = len(seq)
                for pos in range(L + 1):
                    nxt = seq[pos] if pos < L else b
                    tp, ep = tU[prev], eU[prev]
                    F = F0 + tUn[prev] + tUn[nxt] - tp[nxt] + s
                    E = Ea0 + eUn[prev] + eUn[nxt] - ep[nxt] + es
                    if tg > F:
                        E += ph * (tg - F)
                    prev = nxt
                    if E > lim:
                        continue
                    d = (F if F > base else base) - dur
                    dE = E - E0
                    c = d * inv_T + w_th * dE
                    if c < bc_cost:
                        bc_cost = c
                        best_c = (d, dE, c, k, idx, si, pos, F, E, n)
                    if d < bd_d or (d == bd_d and dE < bd_e):
                        bd_d, bd_e = d, dE
                        best_d = (d, dE, c, k, idx, si, pos, F, E, n)
            if best_c is not None:
                res.append(best_c)
                if best_d[5:7] != best_c[5:7]:
                    res.append(best_d)
        return res

    def valid(self, pl):
        k, idx, si = pl[3], pl[4], pl[5]
        if si == -1 and (self.cnt[k] >= self.P.cap[k] or len(self.sorties[k][idx]) >= self.P.R[k]):
            return False
        return self.used[k] + pl[0] <= self.P.T + EPS

    def apply_placement(self, pl):
        k, idx, si, pos, F, E, n = pl[3], pl[4], pl[5], pl[6], pl[7], pl[8], pl[9]
        lst = self.sorties[k][idx]
        if si == -1:
            lst.append({"seq": [n], "F": F, "E": E})
            self.cnt[k] += 1
        else:
            seq = lst[si]["seq"]
            lst[si] = {"seq": seq[:pos] + [n] + seq[pos:], "F": F, "E": E}
        self.refresh_arc(k, idx)

    def best_place(self, n, w, key=None, arcs=None):
        """Vị trí chèn hợp lệ tốt nhất của n trên toàn bộ (hoặc một tập) cung.
        `key`: hàm đánh giá placement (nhỏ hơn là tốt hơn); mặc định là chi phí chèn pl[2]."""
        kf = key or (lambda pl: pl[2])
        best, best_v = None, None
        for k, idx in arcs or self.arcs():
            for pl in self.arc_options(n, k, idx, w):
                if self.valid(pl):
                    v = kf(pl)
                    if best is None or v < best_v:
                        best, best_v = pl, v
        return best


# Cải thiện đường bay trong một sortie (đầu mút cố định)

def _path_len(D, path):
    return sum(D[x][y] for x, y in zip(path, path[1:]))


def _improve_path(D, path):
    """2-opt + or-opt (đoạn 1..3 nút, cả hai chiều) với hai đầu mút cố định."""
    path = list(path)
    n = len(path)
    improved = True
    while improved:
        improved = False
        # 2-opt
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                a, b, c, d = path[i - 1], path[i], path[j], path[j + 1]
                if D[a][c] + D[b][d] - D[a][b] - D[c][d] < -1e-10:
                    path[i: j + 1] = path[i: j + 1][::-1]
                    improved = True
        # or-opt
        for L in (1, 2, 3):
            i = 1
            while i + L <= n - 1:
                seg = path[i: i + L]
                p, q = path[i - 1], path[i + L]
                gain = D[p][seg[0]] + D[seg[-1]][q] - D[p][q]
                rest = path[:i] + path[i + L:]
                best = None
                for j in range(len(rest) - 1):
                    if j == i - 1:
                        continue
                    u, v = rest[j], rest[j + 1]
                    fwd = D[u][seg[0]] + D[seg[-1]][v] - D[u][v]
                    rev = D[u][seg[-1]] + D[seg[0]][v] - D[u][v]
                    add, flip = (fwd, False) if fwd <= rev else (rev, True)
                    if add - gain < -1e-10 and (best is None or add < best[0]):
                        best = (add, j, flip)
                if best is not None:
                    _, j, flip = best
                    ins = seg[::-1] if flip else seg
                    path = rest[: j + 1] + ins + rest[j + 1:]
                    improved = True
                else:
                    i += 1
    return path


# VNS
class VNSSolver:
    SHAKE_NAMES = ["S1_random", "S2_related", "S3_sortie", "S4_road", "S5_large"]
    LS_NAMES = ["L1_intra_sortie", "L2_insert", "L3_relocate", "L4_swap", "L5_road"]

    def __init__(
        self,
        problem: Problem,
        max_iter: int = 500,
        k_max: int = 5,
        q_min: float = 0.05,
        q_max: float = 0.35,
        q_kick: float = 0.5,
        noise: float = 0.2,
        w_energy: float = 0.5,
        patience: int = 60,
        elite_size: int = 5,
        init_tries: int = 20,
        swap_cands: int = 30,
        swap_near: int = 8,
        road_samples: int = 20,
        accept_equal: bool = True,
        t_start: float = 0.03,
        t_end_ratio: float = 0.02,
        reheat: float = 3.0,
        stag_raise: int = 20,
        tabu_size: int = 2000,
        p_forbid: float = 0.5,
        time_limit: float = None,
        seed: int = 42,
        log_every: int = 1,
        verbose: bool = True,
        print_every: int = 10,
    ):
        self.P = problem
        self.max_iter = max_iter
        self.k_max = max(1, k_max)
        self.q_min, self.q_max, self.q_kick = q_min, max(q_min, q_max), q_kick
        self.noise = noise
        self.w = w_energy
        self.patience = max(1, patience)
        self.elite_size = max(1, elite_size)
        self.init_tries = max(1, init_tries)
        self.swap_cands = swap_cands
        self.swap_near = swap_near
        self.road_samples = road_samples
        self.accept_equal = accept_equal
        # chấp nhận lời giải kém hơn kiểu SA: độ giảm t_start * f1_best được nhận với xác suất 0.5 lúc đầu
        self.t_start = t_start
        self.t_end_ratio = t_end_ratio
        self.reheat = max(1.0, reheat)
        self.stag_raise = max(1, stag_raise)
        self.tabu_size = max(1, tabu_size)
        self.p_forbid = p_forbid
        self.n_retry = 0
        self.time_limit = time_limit
        self.seed = seed
        self.log_every = max(1, log_every)
        self.verbose = verbose
        self.print_every = print_every
        self.best = None
        self.restarts = 0
        self.restart_success = 0
        self.shake_stats = {
            n: {"used": 0, "improved": 0, "new_best": 0, "sideways": 0, "worse": 0, "duplicate": 0}
            for n in self.SHAKE_NAMES
        }
        self.ls_stats = {n: 0 for n in self.LS_NAMES}
        self._lopt = set()  # các đường bay (a, seq, b) đã tối ưu cục bộ, tránh chạy lại 2-opt/or-opt
        self._forbid = set()  # nút tạm cấm chèn lại trong VND ngay sau rung lắc
        self._start = time.time()

    
    # Sửa chữa: chèn tham lam các nút chưa phục vụ (có bộ nhớ đệm theo cung)
    
    def _unserved(self, st, exclude=()):
        P = self.P
        served = st.served_set()
        ex = set(exclude) | self._forbid
        return [i for i in P.info_ids if P.reachable[i] and i not in served and i not in ex]

    def repair(self, st: VNSState, pool=None, arcs=None, mode="density", noise=0.0, w=None):
        """mode 'density': mỗi bước chèn nút có profit / chi phí lớn nhất;
        mode 'profit' : duyệt theo lợi nhuận giảm dần, chèn vào vị trí rẻ nhất.
        `arcs` giới hạn các cung được xét. Trả về số nút đã chèn."""
        P = self.P
        w = self.w if w is None else w
        if pool is None:
            pool = self._unserved(st)
        if not pool:
            return 0
        arcs = arcs if arcs is not None else st.arcs()
        opts = {}
        for n in pool:
            d = {}
            for k, idx in arcs:
                o = st.arc_options(n, k, idx, w)
                if o:
                    d[(k, idx)] = o
            if d:
                opts[n] = d

        def jitter():
            return random.uniform(1.0 - noise, 1.0 + noise) if noise > 0 else 1.0

        def best_valid(n):
            best = None
            for lst in opts[n].values():
                for pl in lst:
                    if st.valid(pl):
                        c = pl[2] * jitter()
                        if best is None or c < best[0]:
                            best = (c, pl)
            return best

        def place(pl):
            n, k, idx = pl[9], pl[3], pl[4]
            st.apply_placement(pl)
            opts.pop(n, None)
            for m in list(opts):
                o = st.arc_options(m, k, idx, w)
                if o:
                    opts[m][(k, idx)] = o
                else:
                    opts[m].pop((k, idx), None)
                    if not opts[m]:
                        del opts[m]

        inserted = 0
        if mode == "profit":
            order = sorted(opts, key=lambda n: -P.profit[n] * jitter())
            for n in order:
                if n not in opts:
                    continue
                b = best_valid(n)
                if b is not None:
                    place(b[1])
                    inserted += 1
            return inserted

        while opts:
            best = None
            for n in list(opts):
                b = best_valid(n)
                if b is None:
                    continue
                sc = P.profit[n] / (b[0] + 1e-4)
                if best is None or sc > best[0]:
                    best = (sc, b[1])
            if best is None:
                break
            place(best[1])
            inserted += 1
        return inserted

    def _reinsert_groups(self, st, groups):
        """Chèn lại các sortie bị giải phóng dưới dạng nguyên nhóm (cắt dần nút cuối nếu không vừa)."""
        P = self.P
        served = st.served_set()
        for g in sorted(groups, key=lambda g: -sum(P.profit[i] for i in g)):
            seq = [x for x in g if x not in served]
            while seq:
                plan = st.best_insertion(seq)
                if plan is not None:
                    st.apply_new(plan)
                    st.refresh_arc(plan[1], plan[2])
                    served.update(seq)
                    break
                seq.pop()

    
    # Nước đi trên hành trình xe tải
    
    def _road_moves(self, st):
        K = len(st.nl)
        routes = [st.nl[k][1:-1] for k in range(K)]
        moves = []
        for k in range(K):
            r = routes[k]
            for i in range(len(r) - 1):
                for j in range(i + 1, len(r)):
                    moves.append(("rev", k, i, j))
        for k1 in range(K):
            for i in range(len(routes[k1])):
                for k2 in range(K):
                    if k2 == k1:
                        for p in range(len(routes[k1])):
                            if p != i and p != i - 1:  # p == i-1 trùng với 2-opt hai nút kề nhau
                                moves.append(("rel", k1, i, k2, p))
                    else:
                        for p in range(len(routes[k2]) + 1):
                            moves.append(("rel", k1, i, k2, p))
        for k1 in range(K):
            for k2 in range(k1 + 1, K):
                for i in range(len(routes[k1])):
                    for j in range(len(routes[k2])):
                        moves.append(("swap", k1, i, k2, j))
        return moves

    def _apply_road_move(self, st, mv):
        typ = mv[0]
        if typ == "rev":
            _, k, i, j = mv
            r = list(st.nl[k][1:-1])
            r[i: j + 1] = r[i: j + 1][::-1]
            rel, ks = st.set_route(k, r), [k]
        elif typ == "rel":
            _, k1, i, k2, p = mv
            r1 = list(st.nl[k1][1:-1])
            x = r1.pop(i)
            if k1 == k2:
                r1.insert(p, x)
                rel, ks = st.set_route(k1, r1), [k1]
            else:
                r2 = list(st.nl[k2][1:-1])
                r2.insert(p, x)
                rel = st.set_route(k1, r1) + st.set_route(k2, r2)
                ks = [k1, k2]
        else:
            _, k1, i, k2, j = mv
            r1, r2 = list(st.nl[k1][1:-1]), list(st.nl[k2][1:-1])
            r1[i], r2[j] = r2[j], r1[i]
            rel = st.set_route(k1, r1) + st.set_route(k2, r2)
            ks = [k1, k2]
        rel += self._fix_violation(st, ks)
        return rel, ks

    def _fix_violation(self, st, ks):
        """Nếu xe vượt T do sortie dài trên cung giữ lại: gỡ dần sortie có (lợi nhuận / thời gian tiết kiệm) thấp nhất."""
        P = self.P
        released = []
        for k in ks:
            while st.used[k] > P.T + EPS:
                best = None
                for idx, lst in enumerate(st.sorties[k]):
                    tg = st.tG[k][idx]
                    for si, so in enumerate(lst):
                        if so["F"] <= tg + EPS:
                            continue
                        other = max([tg] + [o["F"] for j, o in enumerate(lst) if j != si])
                        gain = st.dur[k][idx] - other
                        if gain <= EPS:
                            continue
                        sc = sum(P.profit[i] for i in so["seq"]) / gain
                        if best is None or sc < best[0]:
                            best = (sc, idx, si)
                if best is None:
                    break
                released.append(st.remove_sortie(k, best[1], best[2]))
        return released

    
    # Tìm kiếm cục bộ (VND)
    
    def _ls_intra(self, st):
        P = self.P
        D = P.uav_dist
        improved = False
        for k in range(len(st.nl)):
            nl = st.nl[k]
            for idx in range(len(nl) - 1):
                lst = st.sorties[k][idx]
                a, b, tg = nl[idx], nl[idx + 1], st.tG[k][idx]
                changed = False
                for si, so in enumerate(lst):
                    if len(so["seq"]) < 2:
                        continue
                    key = (a, b, tuple(so["seq"]))
                    if key in self._lopt:
                        continue
                    path = [a] + so["seq"] + [b]
                    new = _improve_path(D, path)
                    seq = new[1:-1]
                    if _path_len(D, new) < _path_len(D, path) - 1e-9:
                        F, E = st._cost(seq, _inner_stats(P, seq), a, b, tg)
                        if E <= P.theta + EPS and E < so["E"] - 1e-12:
                            lst[si] = {"seq": seq, "F": F, "E": E}
                            changed = True
                    if len(self._lopt) > 200000:
                        self._lopt.clear()
                    self._lopt.add((a, b, tuple(lst[si]["seq"])))
                if changed:
                    st.refresh_arc(k, idx)
                    improved = True
        return improved

    def _ls_insert(self, st):
        return self.repair(st) > 0

    def _timeout(self):
        return bool(self.time_limit) and time.time() - self._start > self.time_limit

    def _ls_relocate(self, st):
        """Một lượt nén: chuyển lần lượt từng nút sang vị trí / sortie / cung khác nếu tổng thời gian xe giảm
        (hoặc thời gian không đổi và tiêu hao pin giảm). Không làm đổi f1."""
        locs = st.locations()
        nodes = list(locs)
        random.shuffle(nodes)
        improved = False
        for n in nodes:
            if self._timeout():
                break
            k, idx, si = locs[n]
            e_before = sum(so["E"] for so in st.sorties[k][idx])
            t_before = st.used[k]
            snap = st.snapshot(k, idx)
            if st.remove_node(n, (k, idx, si)):
                st.restore(snap)
                continue
            de_rem = sum(so["E"] for so in st.sorties[k][idx]) - e_before
            dt_rem = st.used[k] - t_before
            pl = st.best_place(n, self.w, key=lambda pl: (pl[0], pl[1]))
            if pl is None:
                st.restore(snap)
                continue
            dt, de = pl[0] + dt_rem, pl[1] + de_rem
            if dt < -1e-9 or (dt <= 1e-12 and de < -1e-9):
                st.apply_placement(pl)
                improved = True
                for kk, ii in {(k, idx), (pl[3], pl[4])}:  # cập nhật vị trí các nút trên 2 cung bị đổi
                    for sj, so in enumerate(st.sorties[kk][ii]):
                        for x in so["seq"]:
                            locs[x] = (kk, ii, sj)
            else:
                st.restore(snap)
        return improved

    def _try_exchange(self, st, removes, j):
        P = self.P
        snaps, lost = [], 0
        for i in removes:
            loc = st.locate(i)
            if loc is None:
                break
            snaps.append(st.snapshot(loc[0], loc[1]))
            rel = st.remove_node(i, loc)
            lost += P.profit[i] + sum(P.profit[x] for g in rel for x in g)
        else:
            if P.profit[j] > lost:
                pl = st.best_place(j, self.w)
                if pl is not None:
                    st.apply_placement(pl)
                    return True
        for s in reversed(snaps):
            st.restore(s)
        return False

    def _ls_swap(self, st):
        """Một lượt đổi 1-1 / 2-1: với mỗi nút chưa phục vụ j (lợi nhuận cao trước), gỡ nút đang phục vụ
        gần j có tổng lợi nhuận nhỏ hơn rồi chèn j."""
        P = self.P
        D = P.uav_dist
        served = st.served_set()
        if not served:
            return False
        p_min = min(P.profit[i] for i in served)
        cands = [j for j in self._unserved(st) if P.profit[j] > p_min]
        cands.sort(key=lambda j: -P.profit[j])
        improved = False
        for j in cands[: self.swap_cands]:
            if self._timeout():
                break
            near = sorted((i for i in served if P.profit[i] < P.profit[j]), key=lambda i: D[j][i])
            near = near[: self.swap_near]
            tries = [[i] for i in near] + [
                [a, b] for a, b in itertools.combinations(near[:5], 2) if P.profit[a] + P.profit[b] < P.profit[j]
            ]
            for rem in tries:
                if self._try_exchange(st, rem, j):
                    improved = True
                    served = st.served_set()
                    break
        return improved

    def _insert_ub(self, st, arcs, released):
        """Tổng lợi nhuận các nút chưa phục vụ có ít nhất một cung qua được cận dưới năng lượng / thời gian.
        Nút vừa bị giải phóng được xét trên mọi cung, các nút khác chỉ trên `arcs`."""
        P = self.P
        tU, eU = P.uav_time, P.uav_energy
        rel = {i for g in released for i in g}
        all_arcs = st.arcs()
        ub = 0
        for n in self._unserved(st):
            s = P.service[n]
            for k, idx in (all_arcs if n in rel else arcs):
                a, b = st.nl[k][idx], st.nl[k][idx + 1]
                if eU[a][n] + eU[n][b] + P.p_srv * s > P.theta + EPS:
                    continue
                if st.used[k] + max(0.0, tU[a][n] + tU[n][b] + s - st.dur[k][idx]) > P.T + EPS:
                    continue
                ub += P.profit[n]
                break
        return ub

    def _ls_road(self, st):
        if self.P.num_road == 0:
            return False
        moves = self._road_moves(st)
        if not moves:
            return False
        random.shuffle(moves)
        cur, cur_f1, _, cur_v = st.objective()
        for mv in moves[: self.road_samples]:
            new = st.copy()
            rel, ks = self._apply_road_move(new, mv)
            self._reinsert_groups(new, rel)
            # cận trên: f1 hiện có + lợi nhuận mọi nút chưa phục vụ còn có thể chèn (theo cận dưới pin/thời gian)
            # (khi lời giải hiện tại khả thi, chỉ nước đi có thể làm f1 tăng mới đáng sửa chữa)
            f1_n = new.objective()[1]
            if cur_v <= EPS and f1_n + self._insert_ub(new, new.arcs(ks), rel) <= cur_f1:
                continue
            # chỉ các cung của xe bị thay đổi mới có thể nhận thêm nút; nút bị giải phóng thì xét mọi cung
            self.repair(new, arcs=new.arcs(ks))
            served = new.served_set()
            left = [i for g in rel for i in g if i not in served and i not in self._forbid]
            if left:
                self.repair(new, pool=left)
            if new.objective()[0] > cur + 1e-9:
                st.assign(new)
                return True
        return False

    def _compact(self, st):
        """Nén lời giải (L1 + L3) tới khi không còn nước đi: không đổi f1, chỉ giải phóng tài nguyên.
        Mỗi nước đi giảm chặt (tổng thời gian xe, tổng năng lượng) theo thứ tự từ điển nên vòng lặp dừng."""
        while not self._timeout():
            a = self._ls_intra(st)
            b = self._ls_relocate(st)
            self.ls_stats["L1_intra_sortie"] += int(a)
            self.ls_stats["L3_relocate"] += int(b)
            if not (a or b):
                break
        return st

    def vnd(self, st):
        """VND trên f1: L2 -> L4 -> L5, quay lại L2 mỗi khi f1 tăng; sau mỗi lần tăng thì nén lại lời giải."""
        nbh = [("L2_insert", self._ls_insert), ("L4_swap", self._ls_swap), ("L5_road", self._ls_road)]
        self._compact(st)
        j = 0
        while j < len(nbh):
            if self._timeout():
                break
            name, fn = nbh[j]
            if fn(st):
                self.ls_stats[name] += 1
                self._compact(st)
                j = 0
            else:
                j += 1
        return st

    
    # Rung lắc (shaking)
    
    def _remove_nodes(self, st, nodes):
        removed = set()
        for n in nodes:
            if n in removed:
                continue
            removed.add(n)
            for g in st.remove_node(n):
                removed.update(g)
        return removed

    def _shaw(self, st, q):
        D = self.P.uav_dist
        served = list(st.served_set())
        if not served:
            return set()
        seed_n = random.choice(served)
        order = sorted(served, key=lambda j: D[seed_n][j] * random.uniform(0.8, 1.2))
        return self._remove_nodes(st, order[:q])

    def _remove_random(self, st, q):
        served = list(st.served_set())
        return self._remove_nodes(st, random.sample(served, min(q, len(served))))

    def _remove_sorties(self, st, q):
        P = self.P
        items = [(k, idx, so) for k in range(len(st.nl)) for idx, lst in enumerate(st.sorties[k]) for so in lst]
        items.sort(
            key=lambda t: sum(P.profit[i] for i in t[2]["seq"]) / max(t[2]["E"], 1e-6) * random.uniform(0.5, 1.5)
        )
        removed = set()
        for k, idx, so in items:
            if len(removed) >= q:
                break
            for si, x in enumerate(st.sorties[k][idx]):
                if x is so:
                    removed.update(st.remove_sortie(k, idx, si))
                    break
        return removed

    def _perturb_roads(self, st, n_moves):
        groups = []
        for _ in range(n_moves):
            moves = self._road_moves(st)
            if not moves:
                break
            rel, _ = self._apply_road_move(st, random.choice(moves))
            groups.extend(rel)
        if groups and random.random() < 0.5:
            self._reinsert_groups(st, groups)
        return {i for g in groups for i in g}

    def _destroy(self, st, op, frac):
        n_served = len(st.served_set())
        q = max(1, int(round(frac * max(1, n_served) * random.uniform(0.7, 1.3))))
        has_road = self.P.num_road > 0
        if op == 0:
            return self._remove_random(st, q)
        if op == 1:
            return self._shaw(st, q)
        if op == 2:
            return self._remove_sorties(st, q)
        if op == 3:
            if not has_road:
                return self._shaw(st, q)
            removed = self._perturb_roads(st, random.randint(1, 2))
            return removed | self._shaw(st, max(1, q // 2))
        removed = self._remove_random(st, q)
        if has_road:
            removed |= self._perturb_roads(st, random.randint(1, 3))
        return removed

    def _rebuild(self, st, removed):
        mode = random.choice(("density", "density", "profit"))
        w = random.uniform(0.1, 1.0)
        if removed and random.random() < 0.5:  # ép đổi nút: chèn các nút khác trước
            self.repair(st, pool=self._unserved(st, exclude=removed), mode=mode, noise=self.noise, w=w)
        self.repair(st, mode=mode, noise=self.noise, w=w)

    def shake_vnd(self, st, k, stag=0, forbid=False):
        """Rung lắc (phá hủy + sửa chữa) rồi VND.
        forbid=True: các nút vừa bị gỡ bị cấm chèn lại trong lúc sửa chữa và VND (nếu không, L2/L4
        thường đưa chúng về đúng chỗ cũ và tái tạo y nguyên lời giải hiện tại); sau đó bỏ cấm, chỉ
        chèn tham lam + nén để các nút này vào vị trí mới trong cấu trúc đã thay đổi."""
        # đình trệ càng lâu thì lân cận càng lớn (cộng thêm 0.1 mỗi stag_raise vòng, tối đa 0.8)
        q_max = min(0.8, self.q_max + 0.1 * (stag // self.stag_raise))
        frac = self.q_min + (q_max - self.q_min) * (k - 1) / max(1, self.k_max - 1)
        op = (k - 1) % len(self.SHAKE_NAMES)
        new = st.copy()
        removed = self._destroy(new, op, frac)
        if forbid and removed:
            self._forbid = set(removed)
            try:
                self._rebuild(new, ())
                self.vnd(new)
            finally:
                self._forbid = set()
            if self.repair(new) > 0:
                self._compact(new)
        else:
            self._rebuild(new, removed)
            self.vnd(new)
        return new, op

    def _kick(self, st, fails=0):
        """Khởi động lại: phá hủy mạnh (q_kick, tăng dần theo số lần restart thất bại liên tiếp)
        + thay đổi hành trình xe + sửa chữa có nhiễu."""
        new = st.copy()
        q_kick = min(0.9, self.q_kick * (1.0 + 0.25 * fails))
        q = max(1, int(round(random.uniform(0.6, 1.0) * q_kick * max(1, len(new.served_set())))))
        removed = self._remove_random(new, q // 2) | self._shaw(new, q - q // 2)
        if self.P.num_road > 0:
            removed |= self._perturb_roads(new, random.randint(1, max(1, min(4, self.P.num_road))))
        self._rebuild(new, removed)
        return new

    
    # Khởi tạo, archive ưu tú
    
    def _init_state(self):
        P = self.P
        best_sol, best_fit = None, None
        for _ in range(self.init_tries):
            ind = heuristic_individual(P) if random.random() < 0.7 else random_individual(P)
            sol = decode(ind, P)
            fit = f1_objective(P, sol)
            if best_fit is None or fit[0] > best_fit[0]:
                best_sol, best_fit = sol, fit
        return VNSState.from_solution(P, best_sol)

    def _update_elite(self, elite, st, fit):
        sig = st.signature()
        if any(s == sig for _, s, _ in elite):
            return
        if len(elite) < self.elite_size or fit > elite[-1][0]:
            elite.append((fit, sig, st.copy()))
            elite.sort(key=lambda t: -t[0])
            del elite[self.elite_size:]

    @staticmethod
    def _pick_elite(elite):
        r = random.choices(range(len(elite)), weights=[1.0 / (i + 1) for i in range(len(elite))])[0]
        return elite[r][2]

    #
    def run(self) -> Dict[int, Dict[str, Any]]:
        random.seed(self.seed)
        np.random.seed(self.seed)
        self._start = time.time()
        history = {}

        cur = self.vnd(self._init_state())
        cur_obj = cur.objective()
        best, best_obj = cur.copy(), cur_obj
        best_served = len(best.served_set())
        elite = []
        self._update_elite(elite, cur, cur_obj[0])
        if self.verbose:
            print("Initial solution: fitness={:.2f} f1={}".format(cur_obj[0], cur_obj[1]))

        cur_sig = cur.signature()
        # tabu các lời giải đã đi qua: không nhận lại lời giải cũ -> tránh quay vòng quanh một cực trị địa phương
        tabu = deque([cur_sig], maxlen=self.tabu_size)
        tabu_set = {cur_sig}

        def visit(sig):
            if len(tabu) == tabu.maxlen:
                tabu_set.discard(tabu[0])
            tabu.append(sig)
            tabu_set.add(sig)

        # nhiệt độ: độ giảm t_start * f1 được chấp nhận với xác suất 0.5 ở đầu, nguội dần theo tiến độ
        T0 = -self.t_start * max(1.0, float(best_obj[1])) / math.log(0.5)
        temp, boost = T0, 1.0
        k, stag, since_restart, fails = 1, 0, 0, 0
        epoch_improved = True
        history[0] = self._record(best_obj, best_served, cur_obj[0], k, stag, temp)

        it = 0
        while it < self.max_iter:
            elapsed = time.time() - self._start
            if self.time_limit and elapsed > self.time_limit:
                if self.verbose:
                    print("Time limit reached")
                break
            progress = it / self.max_iter
            if self.time_limit:
                progress = max(progress, elapsed / self.time_limit)
            temp = T0 * (self.t_end_ratio ** progress) * boost
            boost = 1.0 + (boost - 1.0) * 0.98
            it += 1
            event = ""
            forbid = random.random() < self.p_forbid
            new, op = self.shake_vnd(cur, k, since_restart, forbid)
            sig = new.signature()
            if not forbid and (sig == cur_sig or sig in tabu_set):
                # quay về lời giải cũ -> thử lại ngay với nút bị gỡ bị cấm chèn lại
                self.n_retry += 1
                new, op = self.shake_vnd(cur, k, since_restart, True)
                sig = new.signature()
            obj = new.objective()
            name = self.SHAKE_NAMES[op]
            self.shake_stats[name]["used"] += 1

            improved_best = False
            if sig == cur_sig or sig in tabu_set:
                # rung lắc + VND quay về lời giải đã gặp: chuyển sang lân cận lớn hơn
                self.shake_stats[name]["duplicate"] += 1
                k = k + 1 if k < self.k_max else 1
            elif obj[0] > cur_obj[0] + 1e-9:
                cur, cur_obj, cur_sig = new, obj, sig
                visit(sig)
                self.shake_stats[name]["improved"] += 1
                self._update_elite(elite, cur, cur_obj[0])
                k = 1
                if obj[0] > best_obj[0] + 1e-9:
                    best, best_obj = cur.copy(), obj
                    best_served = len(best.served_set())
                    self.shake_stats[name]["new_best"] += 1
                    improved_best = True
                    if not epoch_improved:
                        epoch_improved = True
                        self.restart_success += 1
            else:
                delta = obj[0] - cur_obj[0]
                if delta >= -1e-9:
                    # nước đi ngang: f1 bằng nhau nhưng lời giải khác
                    accept = self.accept_equal
                    stat = "sideways"
                else:
                    # chấp nhận lời giải kém hơn theo xác suất (skewed / annealed VNS)
                    accept = random.random() < math.exp(delta / max(temp, 1e-12))
                    stat = "worse"
                if accept:
                    cur, cur_obj, cur_sig = new, obj, sig
                    visit(sig)
                    self.shake_stats[name][stat] += 1
                    self._update_elite(elite, cur, cur_obj[0])
                k = k + 1 if k < self.k_max else 1

            if improved_best:
                stag = since_restart = fails = 0
            else:
                stag += 1
                since_restart += 1

            if since_restart >= self.patience:
                start = self._pick_elite(elite) if random.random() < 0.7 else best
                cur = self.vnd(self._kick(start, fails))
                cur_obj, cur_sig = cur.objective(), cur.signature()
                visit(cur_sig)
                self._update_elite(elite, cur, cur_obj[0])
                if cur_obj[0] > best_obj[0] + 1e-9:
                    best, best_obj = cur.copy(), cur_obj
                    best_served = len(best.served_set())
                    self.restart_success += 1
                    epoch_improved = True
                    stag = fails = 0
                else:
                    epoch_improved = False
                    fails += 1
                    # hâm nóng lại: restart càng thất bại liên tiếp thì nhiệt độ càng cao
                    boost = 1.0 + (self.reheat - 1.0) * min(1.0, fails / 3.0)
                self.restarts += 1
                k, since_restart = 1, 0
                event = "  [RESTART]"

            if it % self.log_every == 0:
                history[it] = self._record(best_obj, best_served, cur_obj[0], k, stag, temp)
            if self.verbose and self.print_every and (it % self.print_every == 0 or event):
                print(
                    "Iteration {}: best={:.2f} f1={} cur={:.2f} k={} T={:.2f} stag={}{}".format(
                        it, best_obj[0], best_obj[1], cur_obj[0], k, temp, stag, event
                    )
                )

        history[it] = self._record(best_obj, best_served, cur_obj[0], k, stag, temp)
        self.best = best
        return history

    def _record(self, best_obj, best_served, cur_fit, k, stag, temp=0.0):
        fit, f1, f2, V = best_obj
        return {
            "current_fitness": cur_fit,
            "restart_success": self.restart_success,
            "shake_level": k,
            "temperature": temp,
            "fitness": fit,
            "profit": f1,
            "energy": f2,
            "service_rate": best_served / max(1, self.P.num_info),
            "violation": V,
            "stagnation": stag,
            "restarts": self.restarts,
        }

    def neighborhood_stats(self):
        return {"shaking": self.shake_stats, "local_search": self.ls_stats, "duplicate_retries": self.n_retry}



# 
def _run_one(args):
    data_path, s_max, params, seed = args
    problem = Problem(data_path, s_max_default=s_max)
    solver = VNSSolver(problem, seed=seed, **params)
    history = solver.run()
    solution = solver.best.to_solution()
    return history, solution, solver.neighborhood_stats(), solver.restarts, solver.restart_success, seed


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
        print("Cần truyền --filename và/hoặc --subfolder")
    return files


def main():
    """
    Ví dụ:
        python vns.py --filename S042_N10_R_R3.json --subfolder S/N10
        python vns.py --filename S042_N10_R_R3.json
        python vns.py --subfolder S/N10
        python vns.py --subfolder L/N100 --time-limit 60
        python vns.py --subfolder L/N100 --runs 4 --processing-number 4
    """
    parser = argparse.ArgumentParser(description="Run VNS solver for TD-TOP")
    parser.add_argument("--filename", type=str, default=None,
                        help="Tên file JSON instance, vd S042_N10_R_R3.json")
    parser.add_argument("--subfolder", type=str, default=None,
                        help="Thư mục con dưới DATA_ROOT, vd S/N10 hoặc L/N100")
    parser.add_argument("--data-root", type=str, default="../data/data",
                        help="Thư mục dữ liệu gốc (mặc định ../data/data)")
    parser.add_argument("--max-iter", type=int, default=500, help="Số vòng lặp (shake + VND)")
    parser.add_argument("--k-max", type=int, default=5, help="Số lân cận rung lắc")
    parser.add_argument("--q-min", type=float, default=0.05, help="Tỉ lệ nút bị gỡ ở lân cận k = 1")
    parser.add_argument("--q-max", type=float, default=0.35, help="Tỉ lệ nút bị gỡ ở lân cận k = k_max")
    parser.add_argument("--q-kick", type=float, default=0.5, help="Tỉ lệ nút bị gỡ khi khởi động lại")
    parser.add_argument("--noise", type=float, default=0.2, help="Biên độ nhiễu khi sửa chữa sau rung lắc")
    parser.add_argument("--w-energy", type=float, default=0.5,
                        help="Trọng số tiêu hao pin trong điểm ưu tiên khi chèn nút "
                             "(chỉ dẫn hướng vì pin là ràng buộc, không thuộc hàm mục tiêu)")
    parser.add_argument("--no-accept-equal", action="store_true",
                        help="Không nhận nước đi ngang (lời giải khác nhưng cùng f1)")
    parser.add_argument("--t-start", type=float, default=0.03,
                        help="Nhiệt độ đầu: độ giảm t_start * f1 được chấp nhận với xác suất 0.5")
    parser.add_argument("--t-end-ratio", type=float, default=0.02, help="Tỉ lệ nhiệt độ cuối / nhiệt độ đầu")
    parser.add_argument("--reheat", type=float, default=3.0,
                        help="Hệ số tăng nhiệt độ tối đa sau các lần restart không cải thiện")
    parser.add_argument("--stag-raise", type=int, default=20,
                        help="Cứ mỗi N vòng không cải thiện thì q_max tăng thêm 0.1")
    parser.add_argument("--tabu-size", type=int, default=2000, help="Số chữ ký lời giải đã đi qua được ghi nhớ")
    parser.add_argument("--p-forbid", type=float, default=0.5,
                        help="Xác suất cấm chèn lại các nút vừa bị gỡ trong VND sau rung lắc")
    parser.add_argument("--patience", type=int, default=60,
                        help="Số vòng lặp không cải thiện lời giải tốt nhất trước khi restart")
    parser.add_argument("--elite-size", type=int, default=5, help="Kích thước archive lời giải ưu tú")
    parser.add_argument("--init-tries", type=int, default=20, help="Số cá thể giải mã để chọn lời giải khởi tạo")
    parser.add_argument("--swap-cands", type=int, default=30,
                        help="Số nút chưa phục vụ (lợi nhuận cao nhất) được thử trong lân cận swap")
    parser.add_argument("--swap-near", type=int, default=8,
                        help="Số nút đang phục vụ gần nhất được thử đổi với mỗi nút chưa phục vụ")
    parser.add_argument("--road-samples", type=int, default=20,
                        help="Số nước đi trên hành trình xe tải được thử trong mỗi lần L5")
    parser.add_argument("--time-limit", type=float, default=None, help="Giới hạn thời gian mỗi lần chạy (giây)")
    parser.add_argument("--s-max", type=int, default=2, help="Số sortie tối đa/UAV nếu dữ liệu không khai báo")
    parser.add_argument("--runs", type=int, default=1, help="Số bản chạy độc lập (seed, seed+1, ...)")
    parser.add_argument("--processing-number", type=int, default=1,
                        help="Số tiến trình chạy song song các bản độc lập")
    parser.add_argument("--print-every", type=int, default=10, help="In tiến trình mỗi N vòng lặp (khi --runs 1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    result_dir = Path(__file__).parent / "result" / "vns"

    files = collect_files(data_root, args.filename, args.subfolder)
    if not files:
        return

    params = dict(
        max_iter=args.max_iter, k_max=args.k_max, q_min=args.q_min, q_max=args.q_max,
        q_kick=args.q_kick, noise=args.noise, w_energy=args.w_energy, patience=args.patience,
        elite_size=args.elite_size, init_tries=args.init_tries, swap_cands=args.swap_cands,
        swap_near=args.swap_near, road_samples=args.road_samples,
        accept_equal=not args.no_accept_equal, t_start=args.t_start, t_end_ratio=args.t_end_ratio,
        reheat=args.reheat, stag_raise=args.stag_raise, tabu_size=args.tabu_size, p_forbid=args.p_forbid,
        time_limit=args.time_limit,
        verbose=(args.runs == 1), print_every=args.print_every,
    )

    print(f"\n{'=' * 80}\nVNS Solver (TD-TOP)\n{'=' * 80}\n")
    print(f"Found {len(files)} file(s)")

    for data_file, sub in files:
        print(f"\n{'-' * 80}\nProcessing: {sub}/{data_file.name}\n{'-' * 80}")
        try:
            problem = Problem(str(data_file), s_max_default=args.s_max)

            start_time = time.time()
            jobs = [(str(data_file), args.s_max, params, args.seed + r) for r in range(max(1, args.runs))]
            if len(jobs) > 1 and args.processing_number > 1:
                with multiprocessing.Pool(min(args.processing_number, len(jobs))) as pool:
                    outs = pool.map(_run_one, jobs)
            else:
                outs = [_run_one(j) for j in jobs]
            end_time = time.time()

            scored = [(f1_objective(problem, o[1]), o) for o in outs]
            fit, (history, solution, nb_stats, restarts, successes, seed_used) = max(
                scored, key=lambda t: t[0][0]
            )

            result = {
                "time": end_time - start_time,
                "history": history,
                "best_fitness": fit[0],
                "best_profit": fit[1],
                "best_energy": fit[2],
                "best_service_rate": len(solution.served) / max(1, problem.num_info),
                "num_restarts": restarts,
                "num_restart_success": successes,
                "best_seed": seed_used,
                "runs_best_profit": [t[0][1] for t in scored],
                "neighborhood_stats": nb_stats,
                "best_solution": solution_to_dict(problem, solution, fit),
            }
            save_result(result, result_dir / sub / data_file.name)
            print(f"  Completed in {result['time']:.2f} seconds")
            print(f"  Best profit f1: {result['best_profit']:.2f}  (restarts: {result['num_restarts']})")

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\n{'=' * 80}\nVNS experiments completed!\n{'=' * 80}\n")


if __name__ == "__main__":
    main()