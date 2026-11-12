import os as _os, sys as _sys
_MOO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # moo/ (population.py, utils.py)
if _MOO_ROOT not in _sys.path:
    _sys.path.insert(0, _MOO_ROOT)
import multiprocessing
import sys
import os
import numpy as np
from copy import deepcopy
from population import Population, Individual
from metric import cal_hv_front

class MOPSOPopulation(Population):
    def __init__(self, pop_size):
        super().__init__(pop_size)
        self.ParetoFront = []
        self.velocity = []
        self.personal_best = []
        self.personal_best_objectives = []
        self.global_best = None

    def initialize(self):
        self.velocity = [np.zeros_like(ind.chromosome) for ind in self.indivs]
        self.personal_best = [deepcopy(ind) for ind in self.indivs]
        self.personal_best_objectives = [ind.objectives for ind in self.indivs]
        self.update_global_best()

    def update_global_best(self):
        self.ParetoFront = self.fast_nondominated_sort(self.indivs)
        if len(self.ParetoFront) > 0:
            self.global_best = np.random.choice(self.ParetoFront[0])

    def fast_nondominated_sort(self, indi_list):
        ParetoFront = [[]]
        for individual in indi_list:
            individual.domination_count = 0
            individual.dominated_solutions = []
            for other_individual in indi_list:
                if individual.dominates(other_individual):
                    individual.dominated_solutions.append(other_individual)
                elif other_individual.dominates(individual):
                    individual.domination_count += 1
            if individual.domination_count == 0:
                individual.rank = 0
                ParetoFront[0].append(individual)
        i = 0
        while len(ParetoFront[i]) > 0:
            temp = []
            for individual in ParetoFront[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i + 1
                        temp.append(other_individual)
            i = i + 1
            ParetoFront.append(temp)
        return ParetoFront

    def update_personal_best(self):
        for i, individual in enumerate(self.indivs):
            if individual.dominates(self.personal_best[i]):
                self.personal_best[i] = deepcopy(individual)
                self.personal_best_objectives[i] = individual.objectives

    def update_velocity_and_position(self, w, c1, c2):
        for i, individual in enumerate(self.indivs):
            r1 = np.random.rand(len(individual.chromosome))
            r2 = np.random.rand(len(individual.chromosome))
            cognitive = c1 * r1 * (self.personal_best[i].chromosome - individual.chromosome)
            social = c2 * r2 * (self.global_best.chromosome - individual.chromosome)
            self.velocity[i] = w * self.velocity[i] + cognitive + social
            individual.chromosome += self.velocity[i]
            individual.repair()

    def fast_nondominated_sort_crowding_distance(self, indi_list):
        ParetoFront = [[]]
        for individual in indi_list:
            individual.domination_count = 0
            individual.dominated_solutions = []
            for other_individual in indi_list:
                if individual.dominates(other_individual):
                    individual.dominated_solutions.append(other_individual)
                elif other_individual.dominates(individual):
                    individual.domination_count += 1
            if individual.domination_count == 0:
                individual.rank = 0
                ParetoFront[0].append(individual)
        i = 0
        while len(ParetoFront[i]) > 0:
            temp = []
            for individual in ParetoFront[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i + 1
                        temp.append(other_individual)
            i = i + 1
            ParetoFront.append(temp)
        for front in ParetoFront:
            self.calculate_crowding_distance(front)
        return ParetoFront

    def calculate_crowding_distance(self, front):
        if len(front) > 0:
            solutions_num = len(front)
            for individual in front:
                individual.crowding_distance = 0

            for m in range(len(front[0].objectives)):
                front.sort(key=lambda individual: individual.objectives[m])
                front[0].crowding_distance = 10 ** 9
                front[solutions_num - 1].crowding_distance = 10 ** 9
                m_values = [individual.objectives[m] for individual in front]
                scale = max(m_values) - min(m_values)
                if scale == 0: scale = 1
                for i in range(1, solutions_num - 1):
                    front[i].crowding_distance += (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / scale

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

def run_mopso(processing_number, problem, indi_list, pop_size, max_gen, w, c1, c2, cal_fitness):
    print("MOPSO")
    history = {}
    mopso_pop = MOPSOPopulation(pop_size)
    mopso_pop.pre_indi_gen(indi_list)
    pool = multiprocessing.Pool(processing_number)
    arg = [(problem, individual) for individual in mopso_pop.indivs]
    result = pool.starmap(cal_fitness, arg)
    for individual, fitness in zip(mopso_pop.indivs, result):
        individual.objectives = fitness
    mopso_pop.initialize()
    mopso_pop.update_personal_best()
    mopso_pop.update_global_best()

    print("Generation 0: Done")
    Pareto_store = [list(indi.objectives) for indi in mopso_pop.ParetoFront[0]]
    history[0] = Pareto_store

    for gen in range(max_gen):
        Pareto_store = []
        mopso_pop.update_velocity_and_position(w, c1, c2)

        arg = [(problem, individual) for individual in mopso_pop.indivs]
        result = pool.starmap(cal_fitness, arg)
        for individual, fitness in zip(mopso_pop.indivs, result):
            individual.objectives = fitness

        mopso_pop.update_personal_best()
        mopso_pop.update_global_best()
        mopso_pop.natural_selection()

        print(f"Generation {gen + 1}: Done")
        Pareto_store = [list(indi.objectives) for indi in mopso_pop.ParetoFront[0]]
        history[gen + 1] = Pareto_store

    pool.close()
    # print("MOPSO Done: ", cal_hv_front(mopso_pop.ParetoFront[0], np.array([1, 1, 10, 10])))
    return history


if __name__ == "__main__":
    # Chạy trên TD-TOP: python mopso.py --subfolder S/N10 [--filename ...] [--max-gen ...] (xem moo/utils.py)
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from utils import main
    main(algo="mopso")
