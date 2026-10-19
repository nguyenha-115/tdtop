import json
import random
from pathlib import Path
from typing import Dict, List

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from problem import Problem, Solution, Route, Sortie, TruckPlan, Individual

EPS = 1e-9


# Khởi tạo
def random_individual(problem: Problem) -> Individual:
    truck = list(problem.road_ids) + list(problem.delim_ids)
    random.shuffle(truck)
    info = list(problem.info_ids)
    random.shuffle(info)
    # mỗi cá thể có xác suất ngắt sortie riêng -> đa dạng về độ dài sortie
    p_break = random.uniform(0.2, 0.7)
    mask = [1 if random.random() < p_break else 0 for _ in info]
    return Individual([truck, info, mask])


def heuristic_individual(problem: Problem) -> Individual:
    import math

    P = problem
    depot = P.nodes[0].coord
    offset = random.uniform(0, 2 * math.pi)

    def angle(i):
        x, y = P.nodes[i].coord
        return (math.atan2(y - depot[1], x - depot[0]) + offset) % (2 * math.pi)

    roads = sorted(P.road_ids, key=angle)
    K = P.num_trucks
    # chia thành K đoạn liên tiếp, kích thước dao động nhẹ
    cuts = [0] * (K - 1)  # không có nút đường: các đoạn đều rỗng
    if K > 1 and roads:
        base = [round(len(roads) * (j + 1) / K) for j in range(K - 1)]
        cuts = [max(0, min(len(roads), c + random.randint(-1, 1))) for c in base]
        cuts.sort()
    bounds = [0] + cuts + [len(roads)]
    truck = []
    for j in range(K):
        chunk = roads[bounds[j]: bounds[j + 1]]
        cur, ordered = 0, []
        while chunk:
            nxt = min(chunk, key=lambda x: P.truck_dist[cur][x])
            ordered.append(nxt)
            chunk.remove(nxt)
            cur = nxt
        truck.extend(ordered)
        if j < K - 1:
            truck.append(P.delim_ids[j])

    info = sorted(
        P.info_ids, key=lambda i: P.efficiency[i] * random.uniform(0.6, 1.4), reverse=True
    )
    p_break = random.uniform(0.2, 0.6)
    mask = [1 if random.random() < p_break else 0 for _ in info]
    return Individual([truck, info, mask])


def init_population(num_indi, seed, problem, seed_ratio: float = 0.1):
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    n_seed = int(round(num_indi * seed_ratio))
    popu = [heuristic_individual(problem) for _ in range(n_seed)]
    popu += [random_individual(problem) for _ in range(num_indi - n_seed)]
    return popu


def new_immigrant(problem: Problem, heuristic_prob: float = 0.3) -> Individual:
    """Cá thể mới hoàn toàn (immigrant): heuristic với xác suất `heuristic_prob`, còn lại ngẫu nhiên."""
    if random.random() < heuristic_prob:
        return heuristic_individual(problem)
    return random_individual(problem)


# Lai ghép (PMX trên từng hoán vị, mask đi theo gene)
def _pmx(p1, p2, a, b):
    size = len(p1)
    child = [None] * size
    child[a: b + 1] = p1[a: b + 1]
    seg = set(p1[a: b + 1])
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


def _cut_points(size):
    a = random.randint(0, size - 2)
    b = random.randint(a + 1, size - 1)
    return a, b


def crossover(problem: Problem, parent1: Individual, parent2: Individual):
    t1, i1, m1 = parent1.chromosome
    t2, i2, m2 = parent2.chromosome

    # ---- lớp xe tải ----
    if len(t1) >= 2:
        a, b = _cut_points(len(t1))
        ot1 = _pmx(t1, t2, a, b)
        ot2 = _pmx(t2, t1, a, b)
    else:
        ot1, ot2 = list(t1), list(t2)

    # ---- lớp UAV + mask ----
    if len(i1) >= 2:
        a, b = _cut_points(len(i1))
        oi1 = _pmx(i1, i2, a, b)
        oi2 = _pmx(i2, i1, a, b)
        map1 = dict(zip(i1, m1))
        map2 = dict(zip(i2, m2))
        seg1 = set(i1[a: b + 1])
        seg2 = set(i2[a: b + 1])
        # mask đi theo gene: nút nằm trong đoạn được giữ thì lấy mask của cha giữ đoạn đó
        om1 = [map1[v] if v in seg1 else map2[v] for v in oi1]
        om2 = [map2[v] if v in seg2 else map1[v] for v in oi2]
    else:
        oi1, oi2, om1, om2 = list(i1), list(i2), list(m1), list(m2)

    return Individual([ot1, oi1, om1]), Individual([ot2, oi2, om2])


# Đột biến
def _swap_or_insert(seq, mask=None):
    n = len(seq)
    if n < 2:
        return
    i, j = random.sample(range(n), 2)
    if random.random() < 0.5:  # hoán đổi
        seq[i], seq[j] = seq[j], seq[i]
        if mask is not None:
            mask[i], mask[j] = mask[j], mask[i]
    else:  # rút ra chèn lại
        v = seq.pop(i)
        seq.insert(j, v)
        if mask is not None:
            m = mask.pop(i)
            mask.insert(j, m)


def _invert(seq, mask=None):
    n = len(seq)
    if n < 2:
        return
    i, j = sorted(random.sample(range(n), 2))
    seq[i: j + 1] = seq[i: j + 1][::-1]
    if mask is not None:
        mask[i: j + 1] = mask[i: j + 1][::-1]


def mutation(problem: Problem, indi: Individual, strength: int = 1):
    """Mỗi lần đột biến chọn ngẫu nhiên 1 trong 4 toán tử; lặp `strength` lần."""
    truck, info, mask = (list(x) for x in indi.chromosome)
    for _ in range(max(1, strength)):
        op = random.randint(0, 3)
        if op == 0 and info:  # lật mask ngắt sortie
            n_flip = max(1, len(info) // 10)
            for pos in random.sample(range(len(info)), min(n_flip, len(info))):
                mask[pos] = 1 - mask[pos]
        elif op == 1:  # đổi chỗ / chèn lại nút thông tin
            _swap_or_insert(info, mask)
        elif op == 2:  # đổi chỗ / chèn lại trên lớp xe tải
            _swap_or_insert(truck)
        else:  # đảo đoạn
            if random.random() < 0.5:
                _invert(truck)
            else:
                _invert(info, mask)
    return Individual([truck, info, mask])



# Giải mã
def partition(truck_layer, problem: Problem):
    """Cắt lớp xe tải tại ký tự phân cách thành K danh sách nút đường."""
    routes = [[] for _ in range(problem.num_trucks)]
    k = 0
    for g in truck_layer:
        if g >= problem.delim_base:
            k += 1
        else:
            routes[k].append(g)
    return routes


def _inner_stats(P: Problem, seq):
    """Thời gian (bay giữa các nút + phục vụ) và năng lượng (bay + phục vụ), chưa tính 2 chặng phóng/thu hồi."""
    t = e = 0.0
    for x, y in zip(seq, seq[1:]):
        t += P.uav_time[x][y]
        e += P.uav_energy[x][y]
    for x in seq:
        t += P.service[x]
        e += P.p_srv * P.service[x]
    return t, e


class _Planner:
    """Trạng thái chèn sortie vào các cung của hành trình xe tải."""

    def __init__(self, P: Problem, node_lists):
        self.P = P
        self.nl = node_lists
        K = P.num_trucks
        self.tG = [
            [P.truck_time[nl[i]][nl[i + 1]] for i in range(len(nl) - 1)] for nl in node_lists
        ]
        self.dur = [list(t) for t in self.tG]  # thời gian cung (gồm chờ UAV)
        self.sorties = [[[] for _ in t] for t in self.tG]  # dict {seq, F, E}
        self.used = [sum(t) for t in self.tG]
        self.cnt = [0] * K
        self.violation = sum(max(0.0, u - P.T) for u in self.used)

    def _cost(self, seq, inner, a, b, tg):
        P = self.P
        F = inner[0] + P.uav_time[a][seq[0]] + P.uav_time[seq[-1]][b]
        E = (
            inner[1]
            + P.uav_energy[a][seq[0]]
            + P.uav_energy[seq[-1]][b]
            + P.p_hover * max(0.0, tg - F)
        )
        return F, E

    def best_insertion(self, seq):
        """Tạo sortie mới cho dãy `seq` ở cung tốt nhất. Trả về (key, k, idx, seq, F, E) hoặc None."""
        P = self.P
        cands = [seq] if len(seq) == 1 else [seq, seq[::-1]]
        best = None
        for s in cands:
            inner = _inner_stats(P, s)
            for k in range(P.num_trucks):
                if self.cnt[k] >= P.cap[k]:
                    continue
                nl = self.nl[k]
                for idx in range(len(nl) - 1):
                    if len(self.sorties[k][idx]) >= P.R[k]:
                        continue
                    F, E = self._cost(s, inner, nl[idx], nl[idx + 1], self.tG[k][idx])
                    if E > P.theta + EPS:
                        continue
                    delta = max(0.0, F - self.dur[k][idx])
                    if self.used[k] + delta > P.T + EPS:
                        continue
                    key = (delta, E)
                    if best is None or key < best[0]:
                        best = (key, k, idx, s, F, E)
        return best

    def apply_new(self, plan):
        _, k, idx, s, F, E = plan
        self.sorties[k][idx].append({"seq": list(s), "F": F, "E": E})
        new_dur = max(self.dur[k][idx], F)
        self.used[k] += new_dur - self.dur[k][idx]
        self.dur[k][idx] = new_dur
        self.cnt[k] += 1

    def best_merge(self, node):
        """Chèn `node` vào một sortie sẵn có ở vị trí tốt nhất. Trả về (key, k, idx, si, newseq, F, E) hoặc None."""
        P = self.P
        best = None
        for k in range(P.num_trucks):
            nl = self.nl[k]
            for idx in range(len(nl) - 1):
                lst = self.sorties[k][idx]
                if not lst:
                    continue
                a, b, tg = nl[idx], nl[idx + 1], self.tG[k][idx]
                for si, so in enumerate(lst):
                    other = max([o["F"] for j, o in enumerate(lst) if j != si], default=0.0)
                    seq = so["seq"]
                    for pos in range(len(seq) + 1):
                        new = seq[:pos] + [node] + seq[pos:]
                        F, E = self._cost(new, _inner_stats(P, new), a, b, tg)
                        if E > P.theta + EPS:
                            continue
                        new_dur = max(tg, F, other)
                        delta = new_dur - self.dur[k][idx]
                        if self.used[k] + delta > P.T + EPS:
                            continue
                        key = (delta, E)
                        if best is None or key < best[0]:
                            best = (key, k, idx, si, new, F, E)
        return best

    def apply_merge(self, plan):
        _, k, idx, si, new, F, E = plan
        lst = self.sorties[k][idx]
        lst[si] = {"seq": new, "F": F, "E": E}
        new_dur = max(self.tG[k][idx], max(o["F"] for o in lst))
        self.used[k] += new_dur - self.dur[k][idx]
        self.dur[k][idx] = new_dur


def _make_groups(info_layer, mask, P: Problem):
    """Gom lớp UAV thành các sortie ứng viên; sửa lỗi: bỏ nút không khả dụng, ngắt nhóm
    khi cận dưới năng lượng nội tại đã vượt pin."""
    groups, cur, cur_e = [], [], 0.0
    for node, m in zip(info_layer, mask):
        if not P.reachable[node]:
            continue
        if cur:
            add_e = P.uav_energy[cur[-1]][node] + P.p_srv * P.service[node]
        else:
            add_e = P.p_srv * P.service[node]
        if cur and (m == 1 or cur_e + add_e > P.theta):
            groups.append(cur)
            cur, cur_e = [], 0.0
            add_e = P.p_srv * P.service[node]
        cur.append(node)
        cur_e += add_e
    if cur:
        groups.append(cur)
    return groups


def build_solution(P: Problem, node_lists, pl) -> Solution:
    """Dựng lịch cuối cùng (thời điểm đến/rời, sortie) từ trạng thái chèn `pl` (_Planner hoặc lớp con)."""
    plans = []
    sig = []
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
                    Sortie(
                        truck=k,
                        arc_index=idx,
                        launch=a,
                        land=b,
                        nodes=list(so["seq"]),
                        t_launch=t_launch,
                        t_arrival=t_launch + so["F"],
                        t_wait=max(0.0, tg - so["F"]),
                        energy=so["E"],
                    )
                )
            arrival.append(t_launch + tg)
            departure.append(t_launch + pl.dur[k][idx])
        used = len(nl) > 2 or bool(sorties_out)
        plans.append(TruckPlan(k, Route(list(nl), arrival, departure), sorties_out, used))
        sig.append(
            (
                tuple(nl),
                tuple(sorted((s.arc_index, tuple(s.nodes)) for s in sorties_out)),
            )
        )
    return Solution(plans, pl.violation, hash(tuple(sig)))


def decode(indi: Individual, problem: Problem, use_fill: bool = True) -> Solution:
    P = problem
    truck_layer, info_layer, mask = indi.chromosome

    routes = partition(truck_layer, P)
    node_lists = [[0] + r + [P.end] for r in routes]
    pl = _Planner(P, node_lists)

    # ---- chèn các sortie ứng viên theo thứ tự ưu tiên ----
    pending = []
    for g in _make_groups(info_layer, mask, P):
        seq = list(g)
        while seq:
            plan = pl.best_insertion(seq)
            if plan is not None:
                pl.apply_new(plan)
                break
            pending.append(seq.pop())  # cắt nút cuối, thử lại

    # ---- bổ sung nút còn lại (tùy chọn) ----
    if use_fill:
        served = {i for k in range(P.num_trucks) for lst in pl.sorties[k] for so in lst for i in so["seq"]}
        todo = [i for i in P.info_ids if P.reachable[i] and i not in served]
        todo.sort(key=lambda i: -P.profit[i])
        for i in todo:
            opt_new = pl.best_insertion([i])
            opt_merge = pl.best_merge(i)
            k_new = (opt_new[0][0], 1, opt_new[0][1]) if opt_new else None
            k_mer = (opt_merge[0][0], 0, opt_merge[0][1]) if opt_merge else None
            if k_new is None and k_mer is None:
                continue
            if k_mer is not None and (k_new is None or k_mer <= k_new):
                pl.apply_merge(opt_merge)
            else:
                pl.apply_new(opt_new)

    return build_solution(P, node_lists, pl)


# Hàm thích nghi
def cal_objective(problem: Problem, solution: Solution):
    f1 = sum(problem.profit[i] for i in solution.served)
    f2 = solution.energy
    if solution.violation > EPS:
        # vi phạm khung thời gian: luôn thua mọi cá thể khả thi, có độ dốc theo mức vi phạm
        fitness = -(problem.total_profit + 1.0) * (1.0 + solution.violation / problem.T)
    else:
        fitness = f1 - problem.eps * f2
    return fitness, f1, f2


def cal_fitness(problem: Problem, indi_or_chromosome, use_fill: bool = True):
    indi = (
        indi_or_chromosome
        if isinstance(indi_or_chromosome, Individual)
        else Individual(indi_or_chromosome)
    )
    solution = decode(indi, problem, use_fill)
    fitness, f1, f2 = cal_objective(problem, solution)
    service_rate = len(solution.served) / max(1, problem.num_info)
    return fitness, f1, f2, service_rate, solution.violation, solution.signature


# Đo độ đa dạng quần thể
def population_diversity(indivs: List[Individual], n_pairs: int = 40) -> float:
    """Khoảng cách Hamming chuẩn hóa trung bình (lớp xe tải, lớp UAV, mask) trên mẫu các cặp."""
    n = len(indivs)
    if n < 2:
        return 0.0
    total = 0.0
    for _ in range(n_pairs):
        a, b = random.sample(range(n), 2)
        ca, cb = indivs[a].chromosome, indivs[b].chromosome
        d, parts = 0.0, 0
        for la, lb in zip(ca, cb):
            if len(la) == 0:
                continue
            d += sum(1 for x, y in zip(la, lb) if x != y) / len(la)
            parts += 1
        total += d / max(1, parts)
    return total / n_pairs


# kết quả
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
            free_at[m] = s.t_arrival + s.t_wait  # UAV sẵn sàng khi đã về xe
            trips[m] += 1
            sid, uid = f"T{k}-S{n}", f"T{k}-U{m + 1}"
            uav_sorties[m].append(sid)
            used_uavs.add(uid)
            out.append(
                {
                    "id": sid,
                    "truck": k,
                    "uav": uid,
                    "uav_trip_no": trips[m],
                    "launch": oid[s.launch],
                    "land": oid[s.land],
                    "path": [oid[s.launch]] + [oid[i] for i in s.nodes] + [oid[s.land]],
                    "t_launch": s.t_launch,
                    "t_arrival": s.t_arrival,
                    "t_wait": s.t_wait,
                    "energy_wh": s.energy,
                    "served_nodes": [oid[i] for i in s.nodes],
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
                    for m in range(R)
                    if uav_sorties[m]
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


# ====================================================================
# Dữ liệu vào / kết quả ra
# ====================================================================
def get_data_files(base_dir: str) -> Dict[str, list]:
    """Gom các file JSON theo thư mục chứa chúng (vd 'S/N10', 'N20'). Khóa = đường dẫn tương đối."""
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