"""Công cụ cho GA/LNS giải TD-TOP (mã hóa trực tiếp lời giải).

Khác phiên bản cũ (hoán vị + mặt nạ + bộ giải mã tham lam), ở đây một cá thể LÀ một lời giải:
    routes  : với mỗi xe, thứ tự các nút đường
    sorties : với mỗi xe, mỗi cung (a,b) của hành trình, danh sách các sortie (dãy nút thông tin)
Mọi toán tử (chèn / xóa / đổi chỗ nút giữa các sortie, đổi cung của sortie, đổi hành trình xe) đều
thao tác trên cấu trúc này nên GA không còn phải "đoán" lời giải qua bộ giải mã gián tiếp.
Ràng buộc MILP (pin, số sortie/cung, số sortie/xe, khung thời gian) được kiểm tra ngay khi chèn,
nên mọi lời giải sinh ra đều khả thi (trừ khi hành trình xe tải đơn thuần đã vượt T).
"""
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from problem import Problem, Solution, Route, Sortie, TruckPlan  # noqa: E402

EPS = 1e-9
W_T = 200.0  # quy đổi 1 giờ kéo dài thời gian cung ~ 200 Wh khi so sánh vị trí chèn
KNN_SIZE = 18


# ====================================================================
# Tiền xử lý
# ====================================================================
def prepare(P: Problem):
    """Gắn danh sách láng giềng gần nhất (nút thông tin khả dụng) cho mọi nút. Gọi nhiều lần vẫn an toàn."""
    if hasattr(P, "_knn"):
        return
    info = [i for i in P.info_ids if P.reachable[i]]
    knn = {}
    for x in range(P.n_nodes):
        d = sorted(info, key=lambda n: P.uav_dist[x][n])
        knn[x] = [n for n in d if n != x][:KNN_SIZE]
    P._knn = knn
    P._reach_info = info


def metrics(P: Problem, seq, a, b, tg):
    """(F, E, E_fly, E_srv) của sortie a -> seq -> b trên cung có thời gian xe tải tg."""
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


def opt_order(P: Problem, seq, a, b, tg):
    """2-opt + or-opt (cải thiện đầu tiên) cho thứ tự bay trong một sortie, tối thiểu E (+ phạt kéo dài cung)."""
    def J(s):
        F, E, _, _ = metrics(P, s, a, b, tg)
        return E + W_T * max(0.0, F - tg)

    seq = list(seq)
    n = len(seq)
    if n < 2:
        return seq
    cur = J(seq)
    improved = True
    guard = 0
    while improved and guard < 6:
        improved = False
        guard += 1
        # 2-opt (kể cả đảo cả dãy)
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                c = J(cand)
                if c < cur - 1e-9:
                    seq, cur, improved = cand, c, True
        # or-opt: chuyển đoạn 1-2 nút
        for L in (1, 2):
            if n <= L:
                continue
            for i in range(n - L + 1):
                seg = seq[i:i + L]
                rest = seq[:i] + seq[i + L:]
                for j in range(len(rest) + 1):
                    if j == i:
                        continue
                    for s2 in (seg, seg[::-1]) if L > 1 else (seg,):
                        cand = rest[:j] + s2 + rest[j:]
                        c = J(cand)
                        if c < cur - 1e-9:
                            seq, cur, improved = cand, c, True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return seq


# ====================================================================
# Trạng thái lời giải
# ====================================================================
class State:
    """Một lời giải đầy đủ + các thao tác chèn/xóa tăng dần. Tương thích build_solution (tG, dur, sorties, violation)."""

    def __init__(self, P: Problem, routes):
        self.P = P
        self.routes = [list(r) for r in routes]
        self.nl = [[0] + r + [P.end] for r in self.routes]
        tt = P.truck_time
        self.tG = [[tt[nl[i]][nl[i + 1]] for i in range(len(nl) - 1)] for nl in self.nl]
        self.sorties = [[[] for _ in t] for t in self.tG]
        self.dur = [list(t) for t in self.tG]
        self.used = [sum(t) for t in self.tG]
        self.cnt = [0] * P.num_trucks
        self.served = set()
        self.profit = 0
        self.energy = 0.0
        self.violation = sum(max(0.0, u - P.T) for u in self.used)
        self.score = 0.0
        self._rescore()

    # ---------------------------------------------------------------- tiện ích
    def copy(self):
        n = State.__new__(State)
        n.P = self.P
        n.routes = self.routes
        n.nl = self.nl
        n.tG = self.tG
        n.sorties = [
            [[{"seq": list(s["seq"]), "F": s["F"], "E": s["E"], "Ef": s["Ef"], "Es": s["Es"]}
              for s in arc] for arc in tk]
            for tk in self.sorties
        ]
        n.dur = [list(d) for d in self.dur]
        n.used = list(self.used)
        n.cnt = list(self.cnt)
        n.served = set(self.served)
        n.profit = self.profit
        n.energy = self.energy
        n.violation = self.violation
        n.score = self.score
        return n

    def _rescore(self):
        P = self.P
        self.violation = sum(max(0.0, u - P.T) for u in self.used)
        if self.violation > EPS:
            self.score = -(P.total_profit + 1.0) * (1.0 + self.violation / P.T)
        else:
            self.score = self.profit - P.eps * self.energy

    def refresh(self):
        """Tính lại mọi đại lượng tổng hợp từ danh sách sortie (an toàn, rẻ)."""
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
        self._rescore()

    def all_sorties(self):
        return [(k, idx, s) for k in range(self.P.num_trucks)
                for idx, lst in enumerate(self.sorties[k]) for s in lst]

    def clear_touched(self):
        for _, _, s in self.all_sorties():
            s.pop("t", None)

    def signature(self):
        sig = []
        for k in range(self.P.num_trucks):
            sig.append((tuple(self.nl[k]),
                        tuple(sorted((idx, tuple(s["seq"])) for idx, lst in enumerate(self.sorties[k]) for s in lst))))
        return hash(tuple(sig))

    @property
    def fitness(self):
        """(fitness, f1, f2, service_rate, violation, signature) - cùng dạng với bản cũ."""
        P = self.P
        return (self.score, self.profit, self.energy, len(self.served) / max(1, P.num_info),
                self.violation, self.signature())

    def to_gene(self):
        return ([list(r) for r in self.routes],
                [[[list(s["seq"]) for s in lst] for lst in tk] for tk in self.sorties])

    @staticmethod
    def from_gene(P, gene):
        routes, arcs = gene
        st = State(P, routes)
        for k in range(P.num_trucks):
            for idx, lst in enumerate(arcs[k]):
                a, b, tg = st.nl[k][idx], st.nl[k][idx + 1], st.tG[k][idx]
                for seq in lst:
                    F, E, Ef, Es = metrics(P, seq, a, b, tg)
                    st.sorties[k][idx].append({"seq": list(seq), "F": F, "E": E, "Ef": Ef, "Es": Es})
        st.refresh()
        return st

    # ---------------------------------------------------------------- khả năng đặt sortie
    def has_free_slot(self):
        P = self.P
        for k in range(P.num_trucks):
            if self.cnt[k] >= P.cap[k]:
                continue
            for lst in self.sorties[k]:
                if len(lst) < P.R[k]:
                    return True
        return False

    def _arc_options(self, seq, bonus_pair=None):
        """Mọi cách đặt sortie `seq` (hai chiều bay) vào cung còn chỗ. Trả về list (key,k,idx,seq,F,E,Ef,Es)."""
        P = self.P
        out = []
        orients = [seq] if len(seq) == 1 else [seq, seq[::-1]]
        for oi, s in enumerate(orients):
            for k in range(P.num_trucks):
                if self.cnt[k] >= P.cap[k]:
                    continue
                nl = self.nl[k]
                for idx in range(len(nl) - 1):
                    if len(self.sorties[k][idx]) >= P.R[k]:
                        continue
                    a, b, tg = nl[idx], nl[idx + 1], self.tG[k][idx]
                    F, E, Ef, Es = metrics(P, s, a, b, tg)
                    if E > P.theta + EPS:
                        continue
                    delta = max(0.0, F - self.dur[k][idx])
                    if delta > EPS and self.used[k] + delta > P.T + EPS:
                        continue
                    key = E + W_T * delta
                    if bonus_pair is not None and oi == 0 and (a, b) == bonus_pair:
                        key -= 1e6
                    out.append((key, k, idx, s, F, E, Ef, Es))
        return out

    def _attach(self, k, idx, seq, F, E, Ef, Es, touched=True):
        d = {"seq": list(seq), "F": F, "E": E, "Ef": Ef, "Es": Es}
        if touched:
            d["t"] = True
        self.sorties[k][idx].append(d)
        self.cnt[k] += 1
        if F > self.dur[k][idx]:
            self.used[k] += F - self.dur[k][idx]
            self.dur[k][idx] = F
        self.energy += E
        for n in seq:
            self.served.add(n)
            self.profit += self.P.profit[n]
        self._rescore()

    def place_best(self, seq, bonus_pair=None, allow_trim=True):
        """Đặt sortie `seq` vào cung tốt nhất; nếu không được thì bỏ dần nút lợi nhuận thấp nhất."""
        P = self.P
        seq = [n for n in seq if n not in self.served and P.reachable[n]]
        while seq:
            opts = self._arc_options(seq, bonus_pair)
            if opts:
                key, k, idx, s, F, E, Ef, Es = min(opts, key=lambda o: o[0])
                self._attach(k, idx, s, F, E, Ef, Es)
                return True
            if not allow_trim:
                return False
            drop = min(seq, key=lambda n: P.profit[n])
            seq.remove(drop)
        return False

    # ---------------------------------------------------------------- chèn nút vào sortie có sẵn
    def best_insert(self, n, only_touched=False):
        """Vị trí chèn nút n (vào sortie sẵn có) có chi phí năng lượng tăng thêm nhỏ nhất.
        Trả về (cost,k,idx,si,pos,newF,newE,newEf,newEs) hoặc None."""
        P = self.P
        ut, ee = P.uav_time, P.uav_energy
        sn = P.service[n]
        pn = P.p_srv * sn
        th = P.theta + EPS
        ph = P.p_hover
        Tl = P.T + EPS
        best = None
        for k in range(P.num_trucks):
            nl = self.nl[k]
            used = self.used[k]
            for idx, lst in enumerate(self.sorties[k]):
                if not lst:
                    continue
                a, b = nl[idx], nl[idx + 1]
                tg = self.tG[k][idx]
                d0 = self.dur[k][idx]
                for si, s in enumerate(lst):
                    if only_touched and not s.get("t"):
                        continue
                    seq = s["seq"]
                    F, E, Ef, Es = s["F"], s["E"], s["Ef"], s["Es"]
                    oth = tg
                    for j, o in enumerate(lst):
                        if j != si and o["F"] > oth:
                            oth = o["F"]
                    prev = a
                    L = len(seq)
                    for pos in range(L + 1):
                        nxt = seq[pos] if pos < L else b
                        nF = F + ut[prev][n] + ut[n][nxt] - ut[prev][nxt] + sn
                        nEf = Ef + ee[prev][n] + ee[n][nxt] - ee[prev][nxt]
                        nEs = Es + pn
                        hov = tg - nF
                        nE = nEf + nEs + (ph * hov if hov > 0 else 0.0)
                        if nE <= th:
                            delta = (nF if nF > oth else oth) - d0
                            if delta <= EPS or used + delta <= Tl:
                                cost = nE - E + (W_T * delta if delta > 0 else 0.0)
                                if best is None or cost < best[0]:
                                    best = (cost, k, idx, si, pos, nF, nE, nEf, nEs)
                        if pos < L:
                            prev = seq[pos]
        return best

    def apply_insert(self, n, opt):
        _, k, idx, si, pos, nF, nE, nEf, nEs = opt
        s = self.sorties[k][idx][si]
        s["seq"].insert(pos, n)
        self.energy += nE - s["E"]
        s["F"], s["E"], s["Ef"], s["Es"] = nF, nE, nEf, nEs
        s["t"] = True
        d = self.tG[k][idx]
        for o in self.sorties[k][idx]:
            if o["F"] > d:
                d = o["F"]
        self.dur[k][idx] = d
        self.used[k] = sum(self.dur[k])
        self.served.add(n)
        self.profit += self.P.profit[n]
        self._rescore()

    def _near_candidates(self, only_touched):
        P = self.P
        cand = set()
        for k, idx, s in self.all_sorties():
            if only_touched and not s.get("t"):
                continue
            a, b = self.nl[k][idx], self.nl[k][idx + 1]
            for x in [a, b] + s["seq"]:
                cand.update(P._knn[x])
        return [n for n in cand if n not in self.served]

    def greedy_fill(self, rng, noise=0.0, only_touched=False):
        """Chèn lặp nút chưa phục vụ có (lợi nhuận / năng lượng tăng thêm) lớn nhất, có nhiễu."""
        P = self.P
        while True:
            best = None
            for n in self._near_candidates(only_touched):
                opt = self.best_insert(n, only_touched)
                if opt is None:
                    continue
                sc = P.profit[n] / max(opt[0], 0.3)
                if noise:
                    sc *= 1.0 + noise * rng.random()
                if best is None or sc > best[0]:
                    best = (sc, n, opt)
            if best is None:
                break
            self.apply_insert(best[1], best[2])

    def random_fill(self, rng, tries):
        """Chèn thử `tries` nút chưa phục vụ CHỌN NGẪU NHIÊN (mỗi nút vào vị trí rẻ nhất nếu khả thi). Không dùng mật độ."""
        pool = [n for n in self.P._reach_info if n not in self.served]
        rng.shuffle(pool)
        for n in pool[:tries]:
            opt = self.best_insert(n)
            if opt is not None:
                self.apply_insert(n, opt)
            elif self.has_free_slot():
                opts = self._arc_options([n])
                if opts:
                    _, k, idx, s, F, E, Ef, Es = rng.choice(opts)
                    self._attach(k, idx, s, F, E, Ef, Es, touched=False)

    # ---------------------------------------------------------------- xây sortie mới
    def build_sortie(self, rng, trials=4, noise=0.2):
        """Tạo một sortie mới: thử `trials` cặp (cung, nút hạt giống), mở rộng tham lam, giữ sortie lợi nhuận cao nhất."""
        P = self.P
        arcs = []
        for k in range(P.num_trucks):
            if self.cnt[k] >= P.cap[k]:
                continue
            for idx in range(len(self.nl[k]) - 1):
                if len(self.sorties[k][idx]) < P.R[k]:
                    arcs.append((k, idx))
        if not arcs:
            return False
        unserved = [n for n in P._reach_info if n not in self.served]
        if not unserved:
            return False
        best = None
        for _ in range(trials):
            k, idx = rng.choice(arcs)
            a, b = self.nl[k][idx], self.nl[k][idx + 1]
            ud = P.uav_dist
            scored = sorted(unserved, key=lambda n: -P.profit[n] / (ud[a][n] + ud[n][b] + 0.5))
            seed = rng.choice(scored[:6])
            res = self._grow(k, idx, [seed], rng, noise)
            if res is None:
                continue
            seq, F, E, Ef, Es = res
            prof = sum(P.profit[n] for n in seq)
            key = (prof, -E)
            if best is None or key > best[0]:
                best = (key, k, idx, seq)
        if best is None:
            return False
        _, k, idx, seq = best
        a, b, tg = self.nl[k][idx], self.nl[k][idx + 1], self.tG[k][idx]
        seq = opt_order(P, seq, a, b, tg)
        F, E, Ef, Es = metrics(P, seq, a, b, tg)
        self._attach(k, idx, seq, F, E, Ef, Es)
        return True

    def _grow(self, k, idx, seq, rng, noise):
        P = self.P
        ut, ee = P.uav_time, P.uav_energy
        a, b, tg = self.nl[k][idx], self.nl[k][idx + 1], self.tG[k][idx]
        d0, used = self.dur[k][idx], self.used[k]
        F, E, Ef, Es = metrics(P, seq, a, b, tg)
        if E > P.theta + EPS:
            return None
        if F - d0 > EPS and used + (F - d0) > P.T + EPS:
            return None
        seq = list(seq)
        th, Tl, ph = P.theta + EPS, P.T + EPS, P.p_hover
        while True:
            cand = set()
            for x in [a, b] + seq:
                cand.update(P._knn[x])
            best = None
            for n in cand:
                if n in self.served or n in seq:
                    continue
                sn = P.service[n]
                pn = P.p_srv * sn
                prev = a
                L = len(seq)
                for pos in range(L + 1):
                    nxt = seq[pos] if pos < L else b
                    nF = F + ut[prev][n] + ut[n][nxt] - ut[prev][nxt] + sn
                    nEf = Ef + ee[prev][n] + ee[n][nxt] - ee[prev][nxt]
                    nEs = Es + pn
                    hov = tg - nF
                    nE = nEf + nEs + (ph * hov if hov > 0 else 0.0)
                    if nE <= th:
                        delta = max(d0, nF) - d0
                        if delta <= EPS or used + delta <= Tl:
                            cost = nE - E + W_T * delta
                            sc = P.profit[n] / max(cost, 0.3)
                            if noise:
                                sc *= 1.0 + noise * rng.random()
                            if best is None or sc > best[0]:
                                best = (sc, n, pos, nF, nE, nEf, nEs)
                    if pos < L:
                        prev = seq[pos]
            if best is None:
                break
            _, n, pos, F, E, Ef, Es = best
            seq.insert(pos, n)
        return seq, F, E, Ef, Es

    # ---------------------------------------------------------------- xóa / sắp xếp lại
    def remove_nodes(self, nodes):
        P = self.P
        nodes = set(nodes)
        for k in range(P.num_trucks):
            for idx, lst in enumerate(self.sorties[k]):
                if not lst:
                    continue
                a, b, tg = self.nl[k][idx], self.nl[k][idx + 1], self.tG[k][idx]
                newlst = []
                for s in lst:
                    if any(n in nodes for n in s["seq"]):
                        seq = [n for n in s["seq"] if n not in nodes]
                        if seq:
                            F, E, Ef, Es = metrics(P, seq, a, b, tg)
                            if E > P.theta + EPS:  # bớt nút làm tăng thời gian lơ lửng quá pin -> bỏ cả sortie
                                seq = []
                        if seq:
                            s.update(seq=seq, F=F, E=E, Ef=Ef, Es=Es, t=True)
                            newlst.append(s)
                    else:
                        newlst.append(s)
                self.sorties[k][idx] = newlst
        self.refresh()

    def polish(self, rng):
        """Tối ưu thứ tự trong các sortie bị chạm + đặt lại cung cho mọi sortie + chèn thêm nút."""
        P = self.P
        for k, idx, s in self.all_sorties():
            if s.get("t") and len(s["seq"]) > 1:
                a, b, tg = self.nl[k][idx], self.nl[k][idx + 1], self.tG[k][idx]
                new = opt_order(P, s["seq"], a, b, tg)
                F, E, Ef, Es = metrics(P, new, a, b, tg)
                if E <= P.theta + EPS:
                    s.update(seq=new, F=F, E=E, Ef=Ef, Es=Es)
        self.refresh()
        self.reposition(rng)
        self.greedy_fill(rng, 0.0)

    def reposition(self, rng):
        """Với từng sortie, thử đặt sang cung khác (hoặc đảo chiều) nếu giảm năng lượng/thời gian."""
        P = self.P
        items = self.all_sorties()
        rng.shuffle(items)
        for k, idx, s in items:
            lst = self.sorties[k][idx]
            if not any(o is s for o in lst):
                continue
            a, b, tg = self.nl[k][idx], self.nl[k][idx + 1], self.tG[k][idx]
            seq = s["seq"]
            F0, E0, _, _ = metrics(P, seq, a, b, tg)
            # tháo sortie ra
            lst[:] = [o for o in lst if o is not s]
            self.cnt[k] -= 1
            d = tg
            for o in lst:
                d = max(d, o["F"])
            self.used[k] += d - self.dur[k][idx]
            self.dur[k][idx] = d
            cur_key = E0 + W_T * max(0.0, F0 - self.dur[k][idx])
            opts = self._arc_options(seq)
            best = min(opts, key=lambda o: o[0]) if opts else None
            if best is not None and best[0] < cur_key - 1e-7:
                _, k2, idx2, s2, F, E, Ef, Es = best
                self.energy += E - s["E"]
                self.sorties[k2][idx2].append({"seq": list(s2), "F": F, "E": E, "Ef": Ef, "Es": Es, "t": True})
                self.cnt[k2] += 1
                if F > self.dur[k2][idx2]:
                    self.used[k2] += F - self.dur[k2][idx2]
                    self.dur[k2][idx2] = F
            else:  # đặt lại chỗ cũ
                lst.append(s)
                self.cnt[k] += 1
                if s["F"] > self.dur[k][idx]:
                    self.used[k] += s["F"] - self.dur[k][idx]
                    self.dur[k][idx] = s["F"]
        self.refresh()

    # ---------------------------------------------------------------- thay đổi hành trình xe tải
    def with_routes(self, routes, rng):
        """Lời giải mới với hành trình khác: giữ sortie trên các cung (a,b) còn tồn tại, sortie mồ côi được đặt lại."""
        P = self.P
        ns = State(P, routes)
        pairs = {}
        for k in range(P.num_trucks):
            for idx in range(len(ns.nl[k]) - 1):
                pairs.setdefault((ns.nl[k][idx], ns.nl[k][idx + 1]), []).append((k, idx))
        orphans = []
        for k, idx, s in sorted(self.all_sorties(), key=lambda t: -sum(P.profit[n] for n in t[2]["seq"])):
            pair = (self.nl[k][idx], self.nl[k][idx + 1])
            placed = False
            for k2, idx2 in pairs.get(pair, []):
                if ns.cnt[k2] >= P.cap[k2] or len(ns.sorties[k2][idx2]) >= P.R[k2]:
                    continue
                a, b, tg = ns.nl[k2][idx2], ns.nl[k2][idx2 + 1], ns.tG[k2][idx2]
                F, E, Ef, Es = metrics(P, s["seq"], a, b, tg)
                delta = max(0.0, F - ns.dur[k2][idx2])
                if E <= P.theta + EPS and (delta <= EPS or ns.used[k2] + delta <= P.T + EPS):
                    ns._attach(k2, idx2, s["seq"], F, E, Ef, Es, touched=False)
                    placed = True
                    break
            if not placed:
                orphans.append(list(s["seq"]))
        for seq in orphans:
            ns.place_best(seq)
        ns.refresh()
        return ns


# ====================================================================
# Khởi tạo, ruin & recreate, LNS
# ====================================================================
def _split_routes(P, order, rng):
    K = P.num_trucks
    if K == 1:
        return [list(order)]
    cuts = sorted(rng.randint(0, len(order)) for _ in range(K - 1))
    bounds = [0] + cuts + [len(order)]
    return [list(order[bounds[j]:bounds[j + 1]]) for j in range(K)]


def nn_order(P, nodes, start=0):
    nodes, cur, out = list(nodes), start, []
    while nodes:
        nxt = min(nodes, key=lambda x: P.truck_dist[cur][x])
        out.append(nxt)
        nodes.remove(nxt)
        cur = nxt
    return out


def random_routes(P, rng, heuristic=False):
    roads = list(P.road_ids)
    if heuristic:
        depot = P.nodes[0].coord
        off = rng.uniform(0, 2 * math.pi)
        roads.sort(key=lambda i: (math.atan2(P.nodes[i].coord[1] - depot[1], P.nodes[i].coord[0] - depot[0]) + off) % (2 * math.pi))
        K = P.num_trucks
        base = [round(len(roads) * (j + 1) / K) for j in range(K - 1)]
        bounds = [0] + base + [len(roads)]
        return [nn_order(P, roads[bounds[j]:bounds[j + 1]]) for j in range(K)]
    rng.shuffle(roads)
    return _split_routes(P, roads, rng)


def construct(P, routes, rng, trials=4):
    """Dựng lời giải: lần lượt tạo sortie mới tới khi hết chỗ, rồi chèn tham lam, rồi tinh chỉnh."""
    st = State(P, routes)
    guard = 0
    while st.has_free_slot() and guard < sum(P.cap) + 2:
        if not st.build_sortie(rng, trials):
            break
        guard += 1
    st.greedy_fill(rng, 0.0)
    st.polish(rng)
    st.clear_touched()
    return st


def random_construct(P, routes, rng, fill=(0.1, 0.5)):
    """Khởi tạo NGẪU NHIÊN khả thi (không dùng mật độ lợi nhuận, không LNS): duyệt nút theo thứ tự ngẫu nhiên,
    mỗi nút hoặc mở sortie mới trên cung ngẫu nhiên (xác suất p_new ngẫu nhiên theo từng cá thể) hoặc chèn vào sortie
    sẵn có. Dùng cho GA "chuẩn lý thuyết": quần thể ban đầu kém, tiến hóa mới là thứ cải thiện nó."""
    st = State(P, routes)
    nodes = list(P._reach_info)
    rng.shuffle(nodes)
    p_new = rng.uniform(0.05, 0.6)
    q = rng.uniform(*fill)  # "khởi đầu lạnh": mỗi cá thể chỉ thử phục vụ một tỉ lệ q nút -> quần thể đầu xa tối ưu

    def open_new(n):
        opts = st._arc_options([n])
        if not opts:
            return False
        _, k, idx, s, F, E, Ef, Es = rng.choice(opts)
        st._attach(k, idx, s, F, E, Ef, Es, touched=False)
        return True

    for n in nodes:
        if rng.random() > q:
            continue
        if st.has_free_slot() and rng.random() < p_new and open_new(n):
            continue
        opt = st.best_insert(n)
        if opt is not None:
            st.apply_insert(n, opt)
        elif st.has_free_slot():
            open_new(n)
    st.refresh()
    st.clear_touched()
    return st


def ruin(st: State, rng, strength=1.0):
    P = st.P
    served = list(st.served)
    if not served:
        return
    nS = len(served)
    qmax = max(3, int(0.25 * nS * strength))
    q = min(nS, rng.randint(2, qmax))
    r = rng.random()
    if r < 0.2:
        sorts = st.all_sorties()
        nodes = list(rng.choice(sorts)[2]["seq"])
    elif r < 0.4:
        nodes = rng.sample(served, q)
    elif r < 0.7:
        seed = rng.choice(served)
        ud = P.uav_dist[seed]
        nodes = sorted(served, key=lambda n: ud[n] + 0.3 * rng.random())[:q]
    elif r < 0.85:
        nodes = sorted(served, key=lambda n: P.profit[n] / (P.p_srv * P.service[n] + 1.0) * (0.7 + 0.6 * rng.random()))[:q]
    else:
        sorts = st.all_sorties()
        m = min(len(sorts), 2)
        nodes = [n for _, _, s in rng.sample(sorts, m) for n in s["seq"]]
    st.remove_nodes(nodes)


def repair(st: State, rng, trials=3, noise=0.25):
    st.greedy_fill(rng, noise, only_touched=True)
    guard = 0
    while st.has_free_slot() and guard < 3:
        if not st.build_sortie(rng, trials):
            break
        st.greedy_fill(rng, noise, only_touched=True)
        guard += 1


def lns_step(cur: State, rng, tau: float, strength: float = 1.0) -> State:
    new = cur.copy()
    new.clear_touched()
    ruin(new, rng, strength)
    repair(new, rng)
    if new.score > cur.score - max(6.0 * tau, 12.0):  # chỉ tinh chỉnh những ứng viên có triển vọng
        new.polish(rng)
    new.clear_touched()
    return new


def lns(st: State, iters: int, rng, tau0: float = 4.0, tau1: float = 0.3, strength: float = 1.0, deadline=None) -> State:
    """LNS với chấp nhận kiểu SA (nhiệt độ tính theo đơn vị lợi nhuận); trả về lời giải tốt nhất gặp được."""
    import time as _t
    cur = best = st
    for it in range(iters):
        if deadline and _t.time() > deadline:
            break
        tau = tau0 * (tau1 / tau0) ** (it / max(1, iters - 1)) if tau0 > 0 else 0.0
        cand = lns_step(cur, rng, tau, strength)
        d = cand.score - cur.score
        if d >= -EPS or (tau > 0 and rng.random() < math.exp(d / tau)):
            cur = cand
        if cur.score > best.score + EPS:
            best = cur
    return best


# ====================================================================
# Toán tử di truyền
# ====================================================================
def random_route_move(P, routes, rng):
    new = [list(r) for r in routes]
    K = len(new)
    pos = [(k, i) for k in range(K) for i in range(len(new[k]))]
    if not pos:
        return new
    op = rng.randrange(4)
    if op == 0:  # chuyển một nút (có thể sang xe khác)
        k, i = rng.choice(pos)
        v = new[k].pop(i)
        k2 = rng.randrange(K)
        new[k2].insert(rng.randint(0, len(new[k2])), v)
    elif op == 1 and len(pos) >= 2:  # đổi chỗ
        (k1, i1), (k2, i2) = rng.sample(pos, 2)
        new[k1][i1], new[k2][i2] = new[k2][i2], new[k1][i1]
    elif op == 2:  # 2-opt trong một xe
        cand = [k for k in range(K) if len(new[k]) >= 2]
        if cand:
            k = rng.choice(cand)
            i, j = sorted(rng.sample(range(len(new[k])), 2))
            new[k][i:j + 1] = new[k][i:j + 1][::-1]
    else:  # or-opt đoạn 2-3 nút
        cand = [k for k in range(K) if len(new[k]) >= 3]
        if cand:
            k = rng.choice(cand)
            L = min(len(new[k]) - 1, rng.randint(2, 3))
            i = rng.randint(0, len(new[k]) - L)
            seg = new[k][i:i + L]
            del new[k][i:i + L]
            if rng.random() < 0.5:
                seg.reverse()
            k2 = rng.randrange(K)
            j = rng.randint(0, len(new[k2]))
            new[k2][j:j] = seg
    return new


def route_neighbors(P, routes, limit, rng):
    """Liệt kê (một phần) láng giềng của hành trình: relocate, swap, 2-opt (kể cả giữa các xe)."""
    base = [list(r) for r in routes]
    K = len(base)
    key0 = tuple(tuple(r) for r in base)
    res, seen = [], {key0}

    def add(r):
        key = tuple(tuple(x) for x in r)
        if key not in seen:
            seen.add(key)
            res.append(r)

    pos = [(k, i) for k in range(K) for i in range(len(base[k]))]
    for k, i in pos:
        for k2 in range(K):
            for j in range(len(base[k2]) + (0 if k2 == k else 1)):
                r = [list(x) for x in base]
                v = r[k].pop(i)
                r[k2].insert(j, v)
                add(r)
    for a in range(len(pos)):
        for b in range(a + 1, len(pos)):
            r = [list(x) for x in base]
            (k1, i1), (k2, i2) = pos[a], pos[b]
            r[k1][i1], r[k2][i2] = r[k2][i2], r[k1][i1]
            add(r)
    for k in range(K):
        for i in range(len(base[k])):
            for j in range(i + 1, len(base[k])):
                r = [list(x) for x in base]
                r[k][i:j + 1] = r[k][i:j + 1][::-1]
                add(r)
    rng.shuffle(res)
    return res[:limit]


def _pmx(p1, p2, a, b):
    size = len(p1)
    child = [None] * size
    child[a:b + 1] = p1[a:b + 1]
    seg = set(p1[a:b + 1])
    pos2 = {v: i for i, v in enumerate(p2)}
    for i in range(a, b + 1):
        v = p2[i]
        if v in seg:
            continue
        pos = i
        while True:
            pos = pos2[p1[pos]]
            if child[pos] is None:
                break
        child[pos] = v
    for i in range(size):
        if child[i] is None:
            child[i] = p2[i]
    return child


def _to_layer(P, routes):
    layer = []
    for k, r in enumerate(routes):
        layer.extend(r)
        if k < len(routes) - 1:
            layer.append(P.delim_ids[k])
    return layer


def _from_layer(P, layer):
    routes, cur = [], []
    for g in layer:
        if g >= P.delim_base:
            routes.append(cur)
            cur = []
        else:
            cur.append(g)
    routes.append(cur)
    return routes


def crossover(P: Problem, A: State, B: State, rng, light: bool = False) -> State:
    """Lai ghép mức sortie: hành trình = PMX của hai cha mẹ; các sortie của CẢ HAI cha mẹ được nạp vào con theo
    mật độ lợi nhuận/năng lượng (loại nút trùng, ưu tiên cung (a,b) giống cha mẹ), rồi chèn tham lam phần còn lại."""
    la, lb = _to_layer(P, A.routes), _to_layer(P, B.routes)
    if len(la) >= 2 and rng.random() < 0.8:
        i = rng.randint(0, len(la) - 2)
        j = rng.randint(i + 1, len(la) - 1)
        layer = _pmx(la, lb, i, j) if rng.random() < 0.5 else _pmx(lb, la, i, j)
    else:
        layer = list(la if rng.random() < 0.5 else lb)
    child = State(P, _from_layer(P, layer))
    pool = []
    for par in (A, B):
        for k, idx, s in par.all_sorties():
            pair = (par.nl[k][idx], par.nl[k][idx + 1])
            prof = sum(P.profit[n] for n in s["seq"])
            pool.append((prof / max(s["E"], 1.0) * (0.85 + 0.3 * rng.random()), list(s["seq"]), pair))
    pool.sort(key=lambda t: -t[0])
    for _, seq, pair in pool:
        if not child.has_free_slot():
            break
        child.place_best(seq, bonus_pair=pair)
    child.refresh()
    if not light:
        child.greedy_fill(rng, 0.1)
        child.polish(rng)
    child.clear_touched()
    return child


def mutate(P: Problem, st: State, rng, strength: float = 1.0, light: bool = False) -> State:
    if light:  # đột biến ngẫu nhiên thuần: đổi hành trình, hoặc bỏ vài nút rồi chèn ngẫu nhiên vài nút khác
        if rng.random() < 0.3:
            ns = st.with_routes(random_route_move(P, st.routes, rng), rng)
        else:
            ns = st.copy()
            ns.clear_touched()
            nodes = list(ns.served)
            if nodes:
                ns.remove_nodes(rng.sample(nodes, min(len(nodes), rng.randint(1, 3))))
        ns.random_fill(rng, rng.randint(2, 6))
        ns.clear_touched()
        return ns
    if rng.random() < 0.5:
        routes = random_route_move(P, st.routes, rng)
        ns = st.with_routes(routes, rng)
        ns.greedy_fill(rng, 0.2)
        while ns.has_free_slot() and ns.build_sortie(rng, 3):
            ns.greedy_fill(rng, 0.2, only_touched=True)
        if not light:
            ns.polish(rng)
        ns.clear_touched()
        return ns
    ns = st.copy()
    ns.clear_touched()
    for _ in range(max(1, int(round(strength)))):
        ruin(ns, rng, 1.5 + strength)
        repair(ns, rng)
    if not light:
        ns.polish(rng)
    ns.clear_touched()
    return ns


def population_diversity(states: List[State], rng, n_pairs: int = 40) -> float:
    """Khoảng cách Jaccard trung bình giữa các tập nút được phục vụ + khác hành trình (mẫu ngẫu nhiên)."""
    n = len(states)
    if n < 2:
        return 0.0
    tot = 0.0
    for _ in range(n_pairs):
        a, b = rng.sample(range(n), 2)
        sa, sb = states[a].served, states[b].served
        u = len(sa | sb)
        d = 1.0 - (len(sa & sb) / u if u else 1.0)
        d = 0.5 * d + 0.5 * (0.0 if states[a].routes == states[b].routes else 1.0)
        tot += d
    return tot / n_pairs


# ====================================================================
# Giải mã sang Solution (đầu ra tương thích bản cũ) + hàm mục tiêu
# ====================================================================
def build_solution(P: Problem, node_lists, pl) -> Solution:
    plans, sig = [], []
    for k in range(P.num_trucks):
        nl = node_lists[k]
        arrival, departure = [0.0], [0.0]
        sorties_out = []
        for idx in range(len(nl) - 1):
            a, b = nl[idx], nl[idx + 1]
            t_launch = departure[-1]
            tg = pl.tG[k][idx]
            for so in pl.sorties[k][idx]:
                sorties_out.append(
                    Sortie(truck=k, arc_index=idx, launch=a, land=b, nodes=list(so["seq"]),
                           t_launch=t_launch, t_arrival=t_launch + so["F"],
                           t_wait=max(0.0, tg - so["F"]), energy=so["E"])
                )
            arrival.append(t_launch + tg)
            departure.append(t_launch + pl.dur[k][idx])
        used = len(nl) > 2 or bool(sorties_out)
        plans.append(TruckPlan(k, Route(list(nl), arrival, departure), sorties_out, used))
        sig.append((tuple(nl), tuple(sorted((s.arc_index, tuple(s.nodes)) for s in sorties_out))))
    return Solution(plans, pl.violation, hash(tuple(sig)))


def decode(state: State, problem: Problem = None, use_fill=None) -> Solution:
    return build_solution(state.P, state.nl, state)


def cal_objective(problem: Problem, solution: Solution):
    f1 = sum(problem.profit[i] for i in solution.served)
    f2 = solution.energy
    if solution.violation > EPS:
        fitness = -(problem.total_profit + 1.0) * (1.0 + solution.violation / problem.T)
    else:
        fitness = f1 - problem.eps * f2
    return fitness, f1, f2


def verify_solution(P: Problem, solution: Solution):
    """Kiểm tra độc lập các ràng buộc MILP trên Solution. Trả về danh sách lỗi (rỗng = hợp lệ)."""
    errs = []
    seen_road, seen_info = [], set()
    for plan in solution.plans:
        nl = plan.route.nodes
        seen_road += [n for n in nl if P.nodes[n].kind == "road"]
        if nl[0] != 0 or nl[-1] != P.end:
            errs.append("hành trình không bắt đầu/kết thúc ở kho")
        dep = [0.0]
        by_arc = {}
        for s in plan.sorties:
            by_arc.setdefault(s.arc_index, []).append(s)
        if len(plan.sorties) > P.cap[plan.truck]:
            errs.append(f"xe {plan.truck}: quá {P.cap[plan.truck]} sortie")
        for idx in range(len(nl) - 1):
            a, b = nl[idx], nl[idx + 1]
            tg = P.truck_time[a][b]
            lst = by_arc.get(idx, [])
            if len(lst) > P.R[plan.truck]:
                errs.append(f"cung {idx}: quá {P.R[plan.truck]} sortie")
            Fmax = 0.0
            for s in lst:
                path = [a] + s.nodes + [b]
                fly = sum(P.uav_time[x][y] for x, y in zip(path, path[1:]))
                srv = sum(P.service[i] for i in s.nodes)
                F = fly + srv
                E = sum(P.uav_energy[x][y] for x, y in zip(path, path[1:])) + P.p_srv * srv + P.p_hover * max(0.0, tg - F)
                if E > P.theta + 1e-6:
                    errs.append(f"sortie vượt pin: {E:.3f} > {P.theta:.3f}")
                if abs(E - s.energy) > 1e-5:
                    errs.append("năng lượng ghi nhận khác năng lượng tính lại")
                Fmax = max(Fmax, F)
                for i in s.nodes:
                    if i in seen_info:
                        errs.append(f"nút {i} bị phục vụ 2 lần")
                    seen_info.add(i)
            dep.append(dep[-1] + max(tg, Fmax))
        if dep[-1] > P.T + 1e-6 and solution.violation <= EPS:
            errs.append(f"vượt khung thời gian: {dep[-1]:.3f} > {P.T:.3f}")
    if sorted(seen_road) != sorted(P.road_ids):
        errs.append("nút đường không được thăm đúng 1 lần")
    return errs


def solution_to_dict(problem: Problem, solution: Solution, fitness_tuple) -> Dict:
    P = problem
    oid = P.orig_id
    routes = []
    served = solution.served
    total_sorties, used_uavs = 0, set()

    for plan in solution.plans:
        if not plan.used:
            continue
        k = plan.truck + 1
        sorties = sorted(plan.sorties, key=lambda s: (s.t_launch, s.arc_index))
        R = P.R[plan.truck]
        free_at = [0.0] * R
        trips = [0] * R
        uav_sorties = [[] for _ in range(R)]
        out = []
        for n, s in enumerate(sorties, start=1):
            free = [m for m in range(R) if free_at[m] <= s.t_launch + 1e-6]
            m = free[0] if free else min(range(R), key=lambda mm: free_at[mm])
            free_at[m] = s.t_arrival + s.t_wait
            trips[m] += 1
            sid, uid = f"T{k}-S{n}", f"T{k}-U{m + 1}"
            uav_sorties[m].append(sid)
            used_uavs.add(uid)
            out.append(
                {
                    "id": sid, "truck": k, "uav": uid, "uav_trip_no": trips[m],
                    "launch": oid[s.launch], "land": oid[s.land],
                    "path": [oid[s.launch]] + [oid[i] for i in s.nodes] + [oid[s.land]],
                    "t_launch": s.t_launch, "t_arrival": s.t_arrival, "t_wait": s.t_wait,
                    "energy_wh": s.energy, "served_nodes": [oid[i] for i in s.nodes],
                }
            )
        total_sorties += len(out)
        routes.append(
            {
                "id": k,
                "route": [oid[i] for i in plan.route.nodes],
                "arrival": plan.route.arrival,
                "departure": plan.route.departure,
                "uavs": [
                    {"id": f"T{k}-U{m + 1}", "num_sorties": len(uav_sorties[m]), "sorties": uav_sorties[m]}
                    for m in range(R) if uav_sorties[m]
                ],
                "sorties": out,
            }
        )

    return {
        "objective": fitness_tuple[1],
        "f1_total_value": fitness_tuple[1],
        "f2_total_energy_wh": fitness_tuple[2],
        "violation_h": solution.violation,
        "summary": {
            "num_trucks_available": P.num_trucks,
            "num_trucks_used": len(routes),
            "num_info_nodes": P.num_info,
            "num_info_nodes_served": len(served),
            "unserved_info_nodes": [oid[i] for i in P.info_ids if i not in served],
            "total_sorties": total_sorties,
            "total_drones_used": len(used_uavs),
        },
        "routes": routes,
    }


def get_data_files(base_dir: str) -> Dict[str, list]:
    data_files = {}
    base_path = Path(base_dir).resolve()
    if not base_path.exists():
        print(f"Warning: Data directory {base_dir} does not exist!")
        return data_files
    for json_file in sorted(base_path.rglob("*.json")):
        rel = json_file.parent.relative_to(base_path).as_posix()
        data_files.setdefault(rel, []).append(json_file)
    return data_files


def save_result(result: Dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved result to {output_path}")