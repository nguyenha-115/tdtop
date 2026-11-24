import os as _os, sys as _sys
_MOO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # moo/ (population.py, utils.py)
if _MOO_ROOT not in _sys.path:
    _sys.path.insert(0, _MOO_ROOT)
import multiprocessing
import sys
import os
import numpy as np
from population import Population, Individual

# Ghi chú tốc độ: bản cũ cập nhật kho ngoài (external_pop) bằng Individual.dominates trong vòng lặp Python.
# Kho ngoài không giới hạn và giữ cả các lời giải trùng mục tiêu nên lớn tới hàng nghìn phần tử -> mỗi con
# lai phải so trội hàng nghìn lần (~85% thời gian chạy). Bản này giữ thêm ma trận mục tiêu của kho ngoài và so
# trội bằng numpy; natural_selection tính tổng có trọng số cho cả vùng lân cận một lần. Kết quả trùng khớp bản gốc.

def init_weight_vectors_2d(pop_size):
    wvs = []
    for i in np.arange(0, 1 + sys.float_info.epsilon, 1 / (pop_size - 1)):
        wvs.append([i, 1 - i])
    return np.array(wvs)

def init_weight_vectors_3d(pop_size):
    wvs = []
    for i in np.arange(0, 1 + sys.float_info.epsilon, 1 / (pop_size - 1)):
        for j in np.arange(0, 1 + sys.float_info.epsilon, 1 / (pop_size - 1)):
            if i + j <= 1:
                wvs.append([i, j, 1 - i - j])
    return np.array(wvs)

def init_weight_vectors_4d(pop_size):
    wvs = []
    for i in np.arange(0, 1 + sys.float_info.epsilon, 1 / (pop_size - 1)):
        for j in np.arange(0, 1 + sys.float_info.epsilon, 1 / (pop_size - 1)):
            for k in np.arange(0, 1 + sys.float_info.epsilon, 1 / (pop_size - 1)):
                if i + j + k <= 1:
                    wvs.append([i, j, k, 1 - i - j - k])
    return np.array(wvs)


class MOEADPopulation(Population):
    def __init__(self, pop_size,  neighborhood_size, init_weight_vectors):
        super().__init__(pop_size)
        self.neighborhood_size = neighborhood_size
        self.external_pop = []
        self._ext_F = None  # ma trận mục tiêu của external_pop (cùng thứ tự)
        self.weights = init_weight_vectors(self.pop_size)
        self.neighborhoods = self.init_neighborhood()

    def init_neighborhood(self):
        B = np.empty([self.pop_size, self.neighborhood_size], dtype=int)
        for i in range(self.pop_size):
            wv = self.weights[i]
            euclidean_distances = np.empty([self.pop_size], dtype=float)
            for j in range(self.pop_size):
                euclidean_distances[j] = np.linalg.norm(wv - self.weights[j])
            B[i] = np.argsort(euclidean_distances)[:self.neighborhood_size]
        return B

    def reproduction(self, problem, crossover_operator, mutation_operator, mutation_rate):
        offspring = []
        for i in range(self.pop_size):
            parent1, parent2 = np.random.choice(self.neighborhoods[i], 2, replace=False)
            off1, off2 = crossover_operator(problem, self.indivs[parent1], self.indivs[parent2])
            if np.random.rand() < mutation_rate:
                off1 = mutation_operator(problem, off1)
            offspring.append(off1)
        return offspring
    
    # def mutation(self, problem, mutation_operator):
    #     for i in range(self.pop_size):
    #         if np.random.rand() < 0.1:
    #             self.indivs[i] = mutation_operator(problem, self.indivs[i])
    

    def natural_selection(self):
        self.indivs, O = self.indivs[:self.pop_size], self.indivs[self.pop_size:]
        P = np.asarray([ind.objectives for ind in self.indivs], dtype=float)
        for i in range(self.pop_size):
            indi = O[i]
            wv = self.weights[i]
            o = np.asarray(indi.objectives, dtype=float)
            value_indi = np.sum(wv * o)
            nb = self.neighborhoods[i]
            # các j trong vùng lân cận khác nhau nên so sánh với giá trị trước khi thay = vòng lặp gốc
            replace = nb[value_indi < (P[nb] * wv).sum(axis=1)]
            for j in replace:
                self.indivs[j] = indi
            P[replace] = o

    def update_external(self, indivs: list):
        """Như bản gốc: con mới trội phần tử nào thì xóa các phần tử đó và thêm con; nếu không trội ai và
        không bị ai trội thì thêm vào cuối. Thứ tự external_pop giữ nguyên."""
        E = self._ext_F
        if E is None or len(E) != len(self.external_pop):
            E = (np.asarray([ind.objectives for ind in self.external_pop], dtype=float)
                 if self.external_pop else None)
        for indi in indivs:
            o = np.asarray(indi.objectives, dtype=float)
            if E is None:
                E = np.empty((0, len(o)))
            dom = (o <= E).all(axis=1) & (o < E).any(axis=1)  # indi trội phần tử của kho
            if dom.any():
                keep = np.flatnonzero(~dom)
                self.external_pop = [self.external_pop[k] for k in keep]
                E = E[keep]
            elif ((E <= o).all(axis=1) & (E < o).any(axis=1)).any():  # bị trội -> bỏ
                continue
            self.external_pop.append(indi)
            E = np.vstack([E, o])
        self._ext_F = E
    
    # def update_weights(self, problem, indivs: list):
    #     for i in range(self.pop_size):
    #         wv = self.weights[i]
    #         self.indivs[i].objectives = problem.evaluate(indivs[i].chromosome)
    #         value_indi = np.sum(wv * self.indivs[i].objectives)
    #         for j in self.neighborhoods[i]:
    #             if value_indi < np.sum(wv * self.indivs[j].objectives):
    #                 self.indivs[j] = self.indivs[i]


def run_moead(processing_number, problem, indi_list, pop_size, max_gen, neighborhood_size, 
              init_weight_vectors, crossover_operator,mutation_operator, cal_fitness):
    print("MOEA/D")
    moead_pop = MOEADPopulation(pop_size, neighborhood_size, init_weight_vectors)
    moead_pop.pre_indi_gen(indi_list)
    history = {}
    pool = multiprocessing.Pool(processing_number)
    arg = []
    for individual in moead_pop.indivs:
        arg.append((problem, individual))
    result = pool.starmap(cal_fitness, arg)
    for individual, fitness in zip(moead_pop.indivs, result):
        individual.objectives = fitness
    
    moead_pop.update_external(moead_pop.indivs)
    # moead_pop.update_weights(problem, moead_pop.indivs)
    print("Generation 0: Done")
    Pareto_store = []
    for indi in moead_pop.external_pop:
        Pareto_store.append(list(indi.objectives))
    history[0] = Pareto_store

    for gen in range(max_gen):
        offspring = moead_pop.reproduction(problem, crossover_operator, mutation_operator, 0.1)
        arg = []
        for individual in offspring:
            arg.append((problem, individual))
        result = pool.starmap(cal_fitness, arg)
        for individual, fitness in zip(offspring, result):
            individual.objectives = fitness
        moead_pop.update_external(offspring)
        moead_pop.indivs.extend(offspring)
        # moead_pop.update_weights(problem, offspring)
        moead_pop.natural_selection()
        print("Generation {}: Done".format(gen + 1))
        Pareto_store = []
        for indi in moead_pop.external_pop:
            Pareto_store.append(list(indi.objectives))
        history[gen + 1] = Pareto_store
    pool.close()

    # print("MOEA/D Done: ", cal_hv_front(moead_pop.external_pop, np.array([1, 1, 10, 10])))
    return history


if __name__ == "__main__":
    # Chạy trên TD-TOP: python moead.py --subfolder S/N10 [--filename ...] [--max-gen ...] (xem moo/utils.py)
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from utils import main
    main(algo="moead")
