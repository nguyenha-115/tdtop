import os as _os, sys as _sys
_MOO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # moo/ (population.py, utils.py)
if _MOO_ROOT not in _sys.path:
    _sys.path.insert(0, _MOO_ROOT)
import multiprocessing
import os
import sys
from functools import lru_cache
from population import Population, Individual
import random
import numpy as np

# Ghi chú tốc độ: bản cũ (1) sắp xếp không trội bằng vòng lặp Python O(N^2) gọi Individual.dominates,
# (2) gắn `dominated_solutions` / `reference_point` (kèm danh sách `associations`) lên cá thể -> mọi
# deepcopy(parent) trong gen_offspring sao chép cả đồ thị quần thể (và các cá thể đã bị loại ở thế hệ trước),
# (3) tính khoảng cách vuông góc từng cặp (cá thể, điểm tham chiếu) bằng numpy trên list nhỏ,
# (4) import pymoo (metric) không dùng tới. Bản này vector hóa và chỉ lưu chỉ số/số thực lên cá thể;
# phép chọn lọc (kể cả thứ tự front, thứ tự gọi random) giữ nguyên như bản gốc.


@lru_cache(maxsize=None)
def _reference_points(num_objs, num_divisions_per_obj=4):
    """Điểm tham chiếu Das-Dennis, cùng thứ tự với bản đệ quy gốc. Trả về mảng (r, m) (chỉ đọc)."""
    total = num_objs * num_divisions_per_obj
    out = []

    def rec(work, left, depth):
        if depth == num_objs - 1:
            work[depth] = left / total
            out.append(list(work))
            return
        for i in range(left):
            work[depth] = i / total
            rec(work, left - i, depth + 1)

    rec([0.0] * num_objs, total, 0)
    arr = np.asarray(out, dtype=float)
    arr.setflags(write=False)
    return arr


def generate_reference_points(num_objs, num_divisions_per_obj=4):
    return [list(r) for r in _reference_points(num_objs, num_divisions_per_obj)]


def _hyperplane_intercepts(F):
    """Như construct_hyperplane gốc: điểm cực trị = cá thể có mục tiêu o lớn nhất (cá thể cuối nếu hòa)."""
    m = F.shape[1]
    n = F.shape[0]
    ext_idx = [n - 1 - int(np.argmax(F[::-1, o])) for o in range(m)]
    E = F[ext_idx]
    dup = any((E[i] == E[j]).all() for i in range(m) for j in range(i + 1, m))
    if dup:
        return E[np.arange(m), np.arange(m)]
    return 1 / np.linalg.solve(E, np.ones(m))


def niching_select(individuals, k):
    if len(individuals) == k:
        return individuals
    n = len(individuals)
    F = np.asarray([ind.objectives for ind in individuals], dtype=float)  # (n, m)

    # chuẩn hóa (giữ như bản gốc; kết quả chỉ lưu lên cá thể, không dùng cho phép liên kết bên dưới)
    ideal = F.min(axis=0)
    denom = _hyperplane_intercepts(F) - ideal
    denom = np.where(np.abs(denom) > 1e-20, denom, 1e-20)
    norm = F / denom
    for ind, v in zip(individuals, norm.tolist()):
        ind.normalized_values = v

    # liên kết: khoảng cách từ điểm tham chiếu tới đường thẳng qua gốc theo hướng vector mục tiêu (như bản gốc)
    R = _reference_points(F.shape[1])  # (r, m)
    with np.errstate(divide="ignore", invalid="ignore"):
        kk = (F @ R.T) / (F * F).sum(axis=1)[:, None]  # (n, r)
        diff = kk[:, :, None] * F[:, None, :] - R[None, :, :]
        dist = np.sqrt((diff * diff).sum(axis=2))
    best = np.argmin(dist, axis=1)
    best_dist = dist[np.arange(n), best]

    r = len(R)
    assoc = [[] for _ in range(r)]  # chỉ số cá thể theo thứ tự liên kết
    for i, j in enumerate(best.tolist()):
        assoc[j].append(i)
    count = [len(a) for a in assoc]
    alive = list(range(r))  # điểm tham chiếu còn dùng được, giữ thứ tự gốc
    for ind, d in zip(individuals, best_dist.tolist()):
        ind.ref_point_distance = d

    res_idx = []
    while len(res_idx) < k:
        cmin = min(count[j] for j in alive)
        cands = [j for j in alive if count[j] == cmin]
        j = cands[random.randint(0, len(cands) - 1)]
        lst = assoc[j]
        if lst:
            if count[j] == 0:
                pos = min(range(len(lst)), key=lambda p: best_dist[lst[p]])
            else:
                pos = random.randint(0, len(lst) - 1)
            res_idx.append(lst.pop(pos))
            count[j] += 1
        else:
            alive.remove(j)

    res = [individuals[i] for i in res_idx]
    chosen = set(res_idx)
    individuals[:] = [ind for i, ind in enumerate(individuals) if i not in chosen]
    return res


def sel_nsga_iii(individuals, k):
    assert len(individuals) >= k

    if len(individuals) == k:
        return individuals

    fronts = fast_nondominated_sort(individuals)

    limit = 0
    res = []
    for f, front in enumerate(fronts):
        res += front
        if len(res) > k:
            limit = f
            break
    selection = []
    if limit > 0:
        for f in range(limit):
            selection += fronts[f]
    selection += niching_select(fronts[limit], k - len(selection))
    return selection


def fast_nondominated_sort(indi_list):
    """Sắp xếp không trội (vector hóa numpy), thứ tự cá thể trong mỗi front giống bản vòng lặp gốc
    (kể cả front rỗng ở cuối). Không lưu `dominated_solutions` lên cá thể để deepcopy không kéo theo cả quần thể."""
    n = len(indi_list)
    if n == 0:
        return [[]]
    F = np.asarray([ind.objectives for ind in indi_list], dtype=float)
    A, B = F[:, None, :], F[None, :, :]
    D = (A <= B).all(axis=2) & (A < B).any(axis=2)  # D[i, j] <=> i trội j
    count = D.sum(axis=0)

    current = np.flatnonzero(count == 0)
    fronts_idx = [current]
    while len(current) > 0:
        nxt = []
        for p in current:
            qs = np.flatnonzero(D[p])
            if len(qs):
                count[qs] -= 1
                nxt.extend(qs[count[qs] == 0].tolist())
        current = np.asarray(nxt, dtype=int)
        fronts_idx.append(current)

    ParetoFront = []
    for r, idx in enumerate(fronts_idx):
        front = [indi_list[i] for i in idx]
        for ind in front:
            ind.rank = r
            ind.domination_count = 0
            ind.dominated_solutions = None
        ParetoFront.append(front)
    return ParetoFront


class NSGAIIIPopulation(Population):
    def __init__(self, pop_size):
        super().__init__(pop_size)
        self.ParetoFront = []
    def fast_nondominated_sort(self):
        self.ParetoFront = fast_nondominated_sort(self.indivs)

    def natural_selection(self):
        self.indivs = sel_nsga_iii(self.indivs, self.pop_size)


def run_nsga_iii(processing_number, problem, indi_list, pop_size, max_gen, crossover_operator, mutation_operator, 
                crossover_rate, mutation_rate, cal_fitness):
    print("NSGA-III")
    nsga_iii_pop = NSGAIIIPopulation(pop_size)
    nsga_iii_pop.pre_indi_gen(indi_list)

    history = {}
    pool = multiprocessing.Pool(processing_number)
    arg = []
    
    for individual in nsga_iii_pop.indivs:
        arg.append((problem, individual))
    result = pool.starmap(cal_fitness, arg, chunksize=10)

    for individual, fitness in zip(nsga_iii_pop.indivs, result):
        individual.objectives = fitness

    print("Generation 0: Done")

    # dùng lại một pool cho mọi thế hệ (tạo Pool mỗi thế hệ rất tốn, nhất là trên Windows - spawn)
    try:
        for gen in range(max_gen):
            offspring = nsga_iii_pop.gen_offspring(problem, crossover_operator, mutation_operator, crossover_rate, mutation_rate)
            arg = [(problem, individual) for individual in offspring]
            result = pool.starmap(cal_fitness, arg, chunksize=10)
            for individual, fitness in zip(offspring, result):
                individual.objectives = fitness

            nsga_iii_pop.indivs.extend(offspring)
            nsga_iii_pop.natural_selection()
            print(f"Generation {gen+1}: Done")

        # mặt Pareto chỉ cần ở thế hệ cuối
        if max_gen > 0:
            nsga_iii_pop.fast_nondominated_sort()
            history[max_gen] = [list(indi.objectives) for indi in nsga_iii_pop.ParetoFront[0]]
    finally:
        pool.close()

    # print("NSGA-III Done: ", cal_hv_front(nsga_iii_pop.ParetoFront[0], np.array([1,1,10,10])))
    return history


if __name__ == "__main__":
    # Chạy trên TD-TOP: python nsga_iii.py --subfolder S/N10 [--filename ...] [--max-gen ...] (xem moo/utils.py)
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from utils import main
    main(algo="nsga3")
