import os as _os, sys as _sys
_MOO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # moo/ (population.py, utils.py)
if _MOO_ROOT not in _sys.path:
    _sys.path.insert(0, _MOO_ROOT)
import multiprocessing
import sys
import os
import random
import numpy as np
from population import Population, Individual

# Ghi chú tốc độ: bản cũ sắp xếp không trội bằng vòng lặp Python O(N^2) gọi Individual.dominates (~80k lần
# mỗi thế hệ), crowding distance bằng vòng lặp Python, np.random.choice trên list đối tượng (numpy phải dựng
# mảng object mỗi lần gọi) và import pymoo (metric) không dùng. Bản này vector hóa bằng numpy; thứ tự cá thể
# trong các front và chuỗi số ngẫu nhiên giữ nguyên nên kết quả trùng khớp bản gốc.


class MODEPopulation(Population):
    def __init__(self, pop_size):
        super().__init__(pop_size)
        self.ParetoFront = []

    def fast_nondominated_sort_crowding_distance(self, indi_list):
        """Sắp xếp không trội (numpy), thứ tự trong mỗi front giống bản vòng lặp gốc (kể cả front rỗng cuối).
        Không lưu `dominated_solutions` lên cá thể."""
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
            self.calculate_crowding_distance(front)
            ParetoFront.append(front)
        return ParetoFront

    def calculate_crowding_distance(self, front):
        """Crowding distance (numpy). Như bản gốc: `front` được sắp lại tại chỗ lần lượt theo từng mục tiêu
        (stable), nên thứ tự cuối = theo mục tiêu cuối."""
        n = len(front)
        if n == 0:
            return
        F = np.asarray([ind.objectives for ind in front], dtype=float)
        cd = np.zeros(n)
        order = np.arange(n)
        for m in range(F.shape[1]):
            order = order[np.argsort(F[order, m], kind="stable")]
            v = F[order, m]
            cd[order[0]] = 1e9
            cd[order[-1]] = 1e9
            if n > 2:
                scale = v[-1] - v[0]
                if scale == 0:
                    scale = 1
                cd[order[1:-1]] += (v[2:] - v[:-2]) / scale
        front[:] = [front[i] for i in order]
        for ind, d in zip(front, cd[order].tolist()):
            ind.crowding_distance = d

    # Crowding Operator
    def crowding_operator(self, individual, other_individual):
        if (individual.rank < other_individual.rank) or \
                ((individual.rank == other_individual.rank) and (
                        individual.crowding_distance > other_individual.crowding_distance)):
            return 1
        else:
            return -1

    def natural_selection(self):
        self.ParetoFront = self.fast_nondominated_sort_crowding_distance(self.indivs)
        new_indivs = []
        new_fronts = []
        front_num = 0
        while len(new_indivs) + len(self.ParetoFront[front_num]) <= self.pop_size:
            new_indivs.extend(self.ParetoFront[front_num])
            new_fronts.append(self.ParetoFront[front_num])
            if len(new_indivs) == self.pop_size:
                break
            front_num += 1
        self.calculate_crowding_distance(self.ParetoFront[front_num])
        self.ParetoFront[front_num].sort(key=lambda individual: individual.crowding_distance, reverse=True)
        number_remain = self.pop_size - len(new_indivs)
        new_indivs.extend(self.ParetoFront[front_num][0:number_remain])
        new_fronts.append(self.ParetoFront[front_num][0:number_remain])
        self.ParetoFront = new_fronts
        self.indivs = new_indivs

    def gen_offspring_de(self, F, CR):
        offspring = []
        n = len(self.indivs)
        for i in range(self.pop_size):
            # chọn theo chỉ số: cùng lượng số ngẫu nhiên tiêu thụ như np.random.choice(self.indivs, 2, replace=False)
            a, b = np.random.choice(n, 2, replace=False)
            off = rand_1(self.indivs[i], self.indivs[a], self.indivs[b], F, CR)
            offspring.append(off)
        return offspring


def rand_1(indi : Individual, indi1: Individual, indi2: Individual, F, CR):
    x = np.asarray(indi.chromosome, dtype=float)
    trial = np.clip(x + F * (np.asarray(indi1.chromosome) - np.asarray(indi2.chromosome)), -np.pi, np.pi)  # = repair()
    # cùng chuỗi random.random() như vòng lặp gốc (mỗi gen một lần, theo thứ tự)
    keep = np.fromiter((random.random() > CR for _ in range(len(x))), dtype=bool, count=len(x))
    trial[keep] = x[keep]
    return Individual(trial)


def run_mode(processing_number, problem, indi_list, pop_size, max_gen, F, CR, cal_fitness):
    print("MODE")
    history = {}
    mode_pop = MODEPopulation(pop_size)
    mode_pop.pre_indi_gen(indi_list)

    pool = multiprocessing.Pool(processing_number)
    arg = []
    for individual in mode_pop.indivs:
        arg.append((problem, individual))
    result = pool.starmap(cal_fitness, arg)
    for individual, fitness in zip(mode_pop.indivs, result):
        individual.objectives = fitness
    mode_pop.natural_selection()
    print("Generation 0: Done")
    Pareto_store = []
    for indi in mode_pop.ParetoFront[0]:
        Pareto_store.append(list(indi.objectives))
    history[0] = Pareto_store


    for gen in range(max_gen):
        Pareto_store = []
        offspring = mode_pop.gen_offspring_de(F, CR)
        arg = []
        for individual in offspring:
            arg.append((problem, individual))
        result = pool.starmap(cal_fitness, arg)
        for individual, fitness in zip(offspring, result):
            individual.objectives = fitness
        mode_pop.indivs.extend(offspring)
        mode_pop.natural_selection()
        print("Generation {}: Done".format(gen + 1))
        for indi in mode_pop.ParetoFront[0]:
            Pareto_store.append(list(indi.objectives))
        history[gen + 1] = Pareto_store
    pool.close()
    # print("MODE Done: ", cal_hv_front(mode_pop.ParetoFront[0], np.array([1, 1, 10, 10])))
    return history

if __name__ == "__main__":
    # Chạy trên TD-TOP: python mode.py --subfolder S/N10 [--filename ...] [--max-gen ...] (xem moo/utils.py)
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from utils import main
    main(algo="mode")
