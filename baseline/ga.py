import argparse
import multiprocessing
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any

import numpy as np

from problem import Problem, Individual
from utils import (
    init_population,
    new_immigrant,
    cal_fitness,
    decode,
    cal_objective,
    crossover,
    mutation,
    population_diversity,
    solution_to_dict,
    save_result,
)

# ---------------------------------------------------------------------
# Đánh giá song song: mỗi tiến trình giữ sẵn một bản Problem
# ---------------------------------------------------------------------
_WORKER_PROBLEM = None


def _init_worker(problem):
    global _WORKER_PROBLEM
    _WORKER_PROBLEM = problem


def _eval_worker(chromosome):
    return cal_fitness(_WORKER_PROBLEM, chromosome)


class Population:
    def __init__(self, pop_size):
        self.pop_size = pop_size
        self.indivs = []

    def pre_indi_gen(self, indi_list):
        if len(indi_list) != self.pop_size:
            raise ValueError("The length of the list must be equal to the population size")
        self.indivs = deepcopy(indi_list)

    def tournament(self, k):
        """Chọn 1 cá thể bằng giải đấu kích thước k."""
        cands = random.sample(self.indivs, min(k, len(self.indivs)))
        return max(cands, key=lambda ind: ind.fitness[0])

    def gen_offspring(
        self,
        problem,
        crossover_operator,
        mutation_operator,
        crossover_rate,
        mutation_rate,
        tournament_size=3,
        mutation_strength=1,
    ):
        offspring = []
        for _ in range(self.pop_size):
            parent1 = self.tournament(tournament_size)
            parent2 = self.tournament(tournament_size)
            for _try in range(5):  # cố gắng chọn hai cha mẹ khác nhau
                if parent2 is not parent1:
                    break
                parent2 = self.tournament(tournament_size)
            if random.random() < crossover_rate:
                off1, off2 = crossover_operator(problem, parent1, parent2)
            else:
                off1 = Individual(deepcopy(parent1.chromosome))
                off2 = Individual(deepcopy(parent2.chromosome))
            if random.random() < mutation_rate:
                off1 = mutation_operator(problem, off1, mutation_strength)
            if random.random() < mutation_rate:
                off2 = mutation_operator(problem, off2, mutation_strength)
            offspring.append(off1)
            offspring.append(off2)
        return offspring

    def selection(self):
        """(mu+lambda) có loại trùng kiểu hình. Trả về số chỗ trống cần bù bằng cá thể mới."""
        self.indivs.sort(key=lambda ind: ind.fitness[0], reverse=True)
        seen, unique = set(), []
        for ind in self.indivs:
            sig = ind.fitness[5]
            if sig in seen:
                continue
            seen.add(sig)
            unique.append(ind)
            if len(unique) == self.pop_size:
                break
        shortage = self.pop_size - len(unique)
        self.indivs = unique
        return shortage


class GASolver:
    """Genetic Algorithm solver for TD-TOP."""

    def __init__(
        self,
        problem: Problem,
        pop_size: int,
        max_gen: int,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        mutation_rate_max: float = 0.6,
        tournament_size: int = 3,
        immigrant_rate: float = 0.05,
        seed_ratio: float = 0.1,
        patience: int = 15,
        diversity_threshold: float = 0.15,
        elite_keep: float = 0.1,
        time_limit: float = None,
        processing_number: int = 1,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.problem = problem
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_rate_max = max(mutation_rate, mutation_rate_max)
        self.tournament_size = tournament_size
        self.immigrant_rate = immigrant_rate
        self.seed_ratio = seed_ratio
        self.patience = max(1, patience)
        self.diversity_threshold = diversity_threshold
        self.elite_keep = elite_keep
        self.time_limit = time_limit
        self.processing_number = processing_number
        self.seed = seed
        self.verbose = verbose
        self.pool = None
        self.best_individual = None

    # ------------------------------------------------------------------
    def _evaluate(self, indivs):
        if not indivs:
            return
        chroms = [ind.chromosome for ind in indivs]
        if self.pool is not None:
            chunk = max(1, len(chroms) // (4 * self.processing_number))
            result = self.pool.map(_eval_worker, chroms, chunksize=chunk)
        else:
            result = [cal_fitness(self.problem, c) for c in chroms]
        for ind, fit in zip(indivs, result):
            ind.fitness = fit

    def _immigrants(self, n):
        inds = [new_immigrant(self.problem) for _ in range(n)]
        self._evaluate(inds)
        return inds

    def _restart(self, pop: Population):
        """Khởi động lại một phần: giữ tinh hoa, phần còn lại = cá thể mới + cá thể bị đột biến mạnh."""
        pop.indivs.sort(key=lambda ind: ind.fitness[0], reverse=True)
        n_keep = max(2, int(round(self.elite_keep * self.pop_size)))
        elites = pop.indivs[:n_keep]
        n_new = self.pop_size - len(elites)
        n_fresh = n_new // 2
        n_kick = n_new - n_fresh
        fresh = [new_immigrant(self.problem) for _ in range(n_fresh)]
        size = len(elites[0].chromosome[0]) + len(elites[0].chromosome[1])
        strength = max(3, size // 5)
        kicked = [
            mutation(self.problem, random.choice(elites), strength) for _ in range(n_kick)
        ]
        newcomers = fresh + kicked
        self._evaluate(newcomers)
        pop.indivs = elites + newcomers
        pop.selection()  # loại trùng, đồng thời sắp xếp
        short = pop.pop_size - len(pop.indivs)
        if short > 0:
            pop.indivs.extend(self._immigrants(short))
            pop.indivs.sort(key=lambda ind: ind.fitness[0], reverse=True)

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        random.seed(self.seed)
        np.random.seed(self.seed)
        start = time.time()
        history = {}

        indi_list = init_population(self.pop_size, self.seed, self.problem, self.seed_ratio)
        pop = Population(self.pop_size)
        pop.pre_indi_gen(indi_list)
        if self.verbose:
            print("Population initialization")

        if self.processing_number > 1:
            self.pool = multiprocessing.Pool(
                self.processing_number, initializer=_init_worker, initargs=(self.problem,)
            )
        try:
            self._evaluate(pop.indivs)
            pop.selection()
            short = self.pop_size - len(pop.indivs)
            if short > 0:
                pop.indivs.extend(self._immigrants(short))
                pop.indivs.sort(key=lambda ind: ind.fitness[0], reverse=True)

            best = pop.indivs[0]
            best_fit = best.fitness[0]
            self.best_individual = deepcopy(best)
            stagnation, restarts = 0, 0
            div = population_diversity(pop.indivs)
            history[0] = self._record(best, div, self.mutation_rate, 0, restarts)

            n_imm = int(round(self.immigrant_rate * self.pop_size))

            for gen in range(self.max_gen):
                if self.time_limit and time.time() - start > self.time_limit:
                    if self.verbose:
                        print("Time limit reached")
                    break

                # --- đột biến thích nghi: tăng dần khi đình trệ ---
                frac = min(1.0, stagnation / self.patience)
                pm = self.mutation_rate + frac * (self.mutation_rate_max - self.mutation_rate)
                strength = 1 + stagnation // 5

                offspring = pop.gen_offspring(
                    self.problem,
                    crossover,
                    mutation,
                    self.crossover_rate,
                    pm,
                    self.tournament_size,
                    strength,
                )
                offspring.extend(new_immigrant(self.problem) for _ in range(n_imm))
                self._evaluate(offspring)
                pop.indivs.extend(offspring)
                short = pop.selection()
                if short > 0:
                    pop.indivs.extend(self._immigrants(short))
                    pop.indivs.sort(key=lambda ind: ind.fitness[0], reverse=True)

                best = pop.indivs[0]
                if best.fitness[0] > best_fit + 1e-9:
                    best_fit = best.fitness[0]
                    self.best_individual = deepcopy(best)
                    stagnation = 0
                else:
                    stagnation += 1

                div = population_diversity(pop.indivs)
                restarted = False
                if stagnation >= self.patience or div < self.diversity_threshold:
                    if stagnation >= max(3, self.patience // 3):  # tránh restart khi vừa cải thiện
                        self._restart(pop)
                        restarts += 1
                        stagnation = 0
                        restarted = True
                        div = population_diversity(pop.indivs)

                best = pop.indivs[0]
                history[gen + 1] = self._record(best, div, pm, stagnation, restarts)
                if self.verbose:
                    print(
                        "Generation {}: fitness={:.2f} f1={} div={:.3f} pm={:.2f}{}".format(
                            gen + 1,
                            best.fitness[0],
                            best.fitness[1],
                            div,
                            pm,
                            "  [RESTART]" if restarted else "",
                        )
                    )
        finally:
            if self.pool is not None:
                self.pool.close()
                self.pool.join()
                self.pool = None
        return history

    @staticmethod
    def _record(best, diversity, pm, stagnation, restarts):
        fitness, f1, f2, service_rate, violation, _ = best.fitness
        return {
            "fitness": fitness,
            "profit": f1,
            "energy": f2,
            "service_rate": service_rate,
            "violation": violation,
            "diversity": diversity,
            "mutation_rate": pm,
            "stagnation": stagnation,
            "restarts": restarts,
        }


def collect_files(data_root: Path, filename, subfolder):
    """Giống ortool.py: trả về list[(đường dẫn file, thư mục con tương đối)]."""
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
    """Run GA on datasets.

    Ví dụ:
        python ga.py --filename S042_N10_R_R3.json --subfolder S/N10
        python ga.py --filename S042_N10_R_R3.json
        python ga.py --subfolder S/N10
        python ga.py --subfolder L/N100
    """
    parser = argparse.ArgumentParser(description="Run GA solver for TD-TOP")
    parser.add_argument("--filename", type=str, default=None,
                        help="Tên file JSON instance, vd S042_N10_R_R3.json")
    parser.add_argument("--subfolder", type=str, default=None,
                        help="Thư mục con dưới DATA_ROOT, vd S/N10 hoặc L/N100")
    parser.add_argument("--data-root", type=str, default="../data/data",
                        help="Thư mục dữ liệu gốc (mặc định ../data/data, giống ortool.py)")
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--max-gen", type=int, default=100)
    parser.add_argument("--crossover-rate", type=float, default=0.9)
    parser.add_argument("--mutation-rate", type=float, default=0.1)
    parser.add_argument("--mutation-rate-max", type=float, default=0.6)
    parser.add_argument("--tournament-size", type=int, default=5)
    parser.add_argument("--immigrant-rate", type=float, default=0.05)
    parser.add_argument("--seed-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15,
                        help="Số thế hệ không cải thiện trước khi khởi động lại một phần")
    parser.add_argument("--diversity-threshold", type=float, default=0.15)
    parser.add_argument("--elite-keep", type=float, default=0.1)
    parser.add_argument("--time-limit", type=float, default=None, help="Giới hạn thời gian mỗi instance (giây)")
    parser.add_argument("--s-max", type=int, default=2, help="Số sortie tối đa/UAV nếu dữ liệu không khai báo")
    parser.add_argument("--processing-number", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    result_dir = Path(__file__).parent / "result" / "ga"

    files = collect_files(data_root, args.filename, args.subfolder)
    if not files:
        return

    print(f"\n{'=' * 80}\nGA Solver (TD-TOP)\n{'=' * 80}\n")
    print(f"Found {len(files)} file(s)")

    for data_file, sub in files:
        print(f"\n{'-' * 80}\nProcessing: {sub}/{data_file.name}\n{'-' * 80}")
        try:
            problem = Problem(str(data_file), s_max_default=args.s_max)

            start_time = time.time()
            solver = GASolver(
                problem=problem,
                pop_size=args.pop_size,
                max_gen=args.max_gen,
                crossover_rate=args.crossover_rate,
                mutation_rate=args.mutation_rate,
                mutation_rate_max=args.mutation_rate_max,
                tournament_size=args.tournament_size,
                immigrant_rate=args.immigrant_rate,
                seed_ratio=args.seed_ratio,
                patience=args.patience,
                diversity_threshold=args.diversity_threshold,
                elite_keep=args.elite_keep,
                time_limit=args.time_limit,
                processing_number=args.processing_number,
                seed=args.seed,
            )
            history = solver.run()
            end_time = time.time()

            best = solver.best_individual
            solution = decode(best, problem)
            fit = cal_objective(problem, solution)
            last = history[max(history.keys())]

            result = {
                "time": end_time - start_time,
                "history": history,
                "best_fitness": fit[0],
                "best_profit": fit[1],
                "best_energy": fit[2],
                "best_service_rate": len(solution.served) / max(1, problem.num_info),
                "num_restarts": last["restarts"],
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

    print(f"\n{'=' * 80}\nGA experiments completed!\n{'=' * 80}\n")


if __name__ == "__main__":
    main()