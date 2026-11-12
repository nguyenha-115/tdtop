import os as _os, sys as _sys
_MOO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # moo/ (population.py, utils.py)
if _MOO_ROOT not in _sys.path:
    _sys.path.insert(0, _MOO_ROOT)
import multiprocessing
import sys
import os
import numpy as np
from population import Population, Individual
from metric import cal_hv_front

class NSGAIIPopulation(Population):
    def __init__(self, pop_size):
        super().__init__(pop_size)
        self.ParetoFront = []
    

    def fast_nondominated_sort_crowding_distance(self, indi_list):
        """Sắp xếp không trội (vector hóa numpy). Thứ tự cá thể trong mỗi front giống hệt bản vòng lặp gốc.
        Không lưu `dominated_solutions` trên cá thể: tham chiếu chéo đó khiến deepcopy (gen_offspring)
        sao chép cả quần thể."""
        n = len(indi_list)
        if n == 0:
            return [[]]
        F = np.asarray([ind.objectives for ind in indi_list], dtype=float)
        # D[i, j] = True <=> i trội j  (<= mọi mục tiêu và < ít nhất một mục tiêu)
        A, B = F[:, None, :], F[None, :, :]
        D = (A <= B).all(axis=2) & (A < B).any(axis=2)
        count = D.sum(axis=0)  # số cá thể trội j

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
            self.calculate_crowding_distance(front)
            ParetoFront.append(front)
        return ParetoFront

    def calculate_crowding_distance(self, front):
        """Crowding distance (numpy). Như bản gốc: `front` được sắp xếp lại tại chỗ theo mục tiêu cuối (stable)."""
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


def run_nsga_ii(processing_number, problem, indi_list, pop_size, max_gen, crossover_operator, mutation_operator, 
                crossover_rate, mutation_rate, cal_fitness):
    print("NSGA-II")
    history = {}
    nsga_ii_pop = NSGAIIPopulation(pop_size)
    nsga_ii_pop.pre_indi_gen(indi_list)

    pool = multiprocessing.Pool(processing_number)
    arg = []
    for individual in nsga_ii_pop.indivs:
        arg.append((problem, individual))
    result = pool.starmap(cal_fitness, arg)
    for individual, fitness in zip(nsga_ii_pop.indivs, result):
        individual.objectives = fitness
    nsga_ii_pop.natural_selection()
    print("Generation 0: Done")
    Pareto_store = []
    for indi in nsga_ii_pop.ParetoFront[0]:
        Pareto_store.append(list(indi.objectives))
    history[0] = Pareto_store


    for gen in range(max_gen):
        Pareto_store = []
        offspring = nsga_ii_pop.gen_offspring(problem, crossover_operator, mutation_operator, crossover_rate, mutation_rate)
        arg = []
        for individual in offspring:
            arg.append((problem, individual))
        result = pool.starmap(cal_fitness, arg)
        for individual, fitness in zip(offspring, result):
            individual.objectives = fitness
        nsga_ii_pop.indivs.extend(offspring)
        nsga_ii_pop.natural_selection()
        print("Generation {}: Done".format(gen + 1))
        for indi in nsga_ii_pop.ParetoFront[0]:
            Pareto_store.append(list(indi.objectives))
        history[gen + 1] = Pareto_store
    pool.close()
    # print("NSGA-II Done: ", cal_hv_front(nsga_ii_pop.ParetoFront[0], np.array([1, 1, 10, 10])))
    return history
    



if __name__ == "__main__":
    # Chạy trên TD-TOP: python nsga_ii.py --subfolder S/N10 [--filename ...] [--max-gen ...] (xem moo/utils.py)
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from utils import main
    main(algo="nsga2")
