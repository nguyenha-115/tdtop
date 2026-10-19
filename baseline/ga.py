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

# Đánh giá song song: mỗi tiến trình giữ sẵn một bản Problem
_WORKER_PROBLEM = None
_WORKER_USE_FILL = True


def _init_worker(problem, use_fill=True):
    global _WORKER_PROBLEM, _WORKER_USE_FILL
    _WORKER_PROBLEM = problem
    _WORKER_USE_FILL = use_fill


def _eval_worker(chromosome):
    return cal_fitness(_WORKER_PROBLEM, chromosome, _WORKER_USE_FILL)


class Population:
    def __init__(self, pop_size):
        self.pop_size = pop_size
        self.indivs = []

    def pre_indi_gen(self, indi_list):
        if len(indi_list) != self.pop_size:
            raise ValueError("The length of the list must be equal to the population size")
        self.indivs = deepcopy(indi_list)

    def tournament(self, k):
        """Chọn 1 cá thể bằng kích thước k."""
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
    def __init__(
        self,
        problem: Problem,
        pop_size: int,
        max_gen: int,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.3,
        mutation_rate_max: float = 0.8,
        tournament_size: int = 2,
        immigrant_rate: float = 0.1,
        seed_ratio: float = 0.2,
        patience: int = 30,
        converge_threshold: float = 0.002,
        archive_size: int = 8,
        isolation: int = 10,
        max_shake: int = 4,
        restart_fresh: float = 0.3,
        restart_strength: float = 0.04,
        use_fill: bool = False,
        heuristic_prob: float = 0.3,
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
        self.converge_threshold = converge_threshold
        self.archive_size = max(1, archive_size)
        self.isolation = max(1, isolation)
        self.max_shake = max(1, max_shake)
        self.archive = []
        self.restart_fresh = restart_fresh
        self.restart_strength = restart_strength
        self.use_fill = use_fill
        self.heuristic_prob = heuristic_prob
        self.time_limit = time_limit
        self.processing_number = processing_number
        self.seed = seed
        self.verbose = verbose
        self.pool = None
        self.best_individual = None

    def _evaluate(self, indivs):
        if not indivs:
            return
        chroms = [ind.chromosome for ind in indivs]
        if self.pool is not None:
            chunk = max(1, len(chroms) // (4 * self.processing_number))
            result = self.pool.map(_eval_worker, chroms, chunksize=chunk)
        else:
            result = [cal_fitness(self.problem, c, self.use_fill) for c in chroms]
        for ind, fit in zip(indivs, result):
            ind.fitness = fit

    def _immigrants(self, n):
        inds = [new_immigrant(self.problem, self.heuristic_prob) for _ in range(n)]
        self._evaluate(inds)
        return inds

    def _update_archive(self, cands):
        sigs = {a.fitness[5] for a in self.archive}
        for ind in cands:
            if ind.fitness[5] not in sigs:
                self.archive.append(deepcopy(ind))
                sigs.add(ind.fitness[5])
        self.archive.sort(key=lambda ind: ind.fitness[0], reverse=True)
        del self.archive[self.archive_size:]

    def _restart(self, pop: Population, shake_level):
        hard = shake_level > self.max_shake
        size = len(self.archive[0].chromosome[0]) + len(self.archive[0].chromosome[1])
        base = max(1, int(round(self.restart_strength * size)))
        fresh_ratio = max(self.restart_fresh, 0.8) if hard else self.restart_fresh
        n_fresh = int(round(fresh_ratio * self.pop_size))
        n_kick = self.pop_size - n_fresh
        newcomers = [new_immigrant(self.problem, self.heuristic_prob) for _ in range(n_fresh)]
        lo = base * (shake_level - 1) + 1 if not hard else base * self.max_shake
        hi = base * shake_level if not hard else base * (self.max_shake + 1)
        for _ in range(n_kick):
            # một nửa đá quanh tốt nhất toàn cục, nửa còn lại quanh các cực trị khác trong archive
            src = self.archive[0] if random.random() < 0.5 else random.choice(self.archive)
            newcomers.append(mutation(self.problem, src, random.randint(lo, max(lo, hi))))
        self._evaluate(newcomers)
        # cá thể đá xong vẫn trùng kiểu hình với archive thì loại -> epoch thực sự cô lập
        known = {a.fitness[5] for a in self.archive}
        pop.indivs = [ind for ind in newcomers if ind.fitness[5] not in known]
        short = pop.selection()
        if short > 0:
            pop.indivs.extend(self._immigrants(short))
            pop.indivs.sort(key=lambda ind: ind.fitness[0], reverse=True)

    def _reinject(self, pop: Population):
        """Kết thúc giai đoạn cô lập: đưa archive trở lại, thay các cá thể kém nhất."""
        inject = [deepcopy(a) for a in self.archive]
        pop.indivs = pop.indivs[: self.pop_size - len(inject)] + inject
        pop.selection()
        short = self.pop_size - len(pop.indivs)
        if short > 0:
            pop.indivs.extend(self._immigrants(short))
            pop.indivs.sort(key=lambda ind: ind.fitness[0], reverse=True)

    def _converged(self, pop: Population):
        top = pop.indivs[0].fitness[0]
        mid = pop.indivs[len(pop.indivs) // 2].fitness[0]
        return (top - mid) / max(1.0, abs(top)) < self.converge_threshold

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
                self.processing_number, initializer=_init_worker, initargs=(self.problem, self.use_fill)
            )
        try:
            self._evaluate(pop.indivs)
            pop.selection()
            short = self.pop_size - len(pop.indivs)
            if short > 0:
                pop.indivs.extend(self._immigrants(short))
                pop.indivs.sort(key=lambda ind: ind.fitness[0], reverse=True)

            self.archive = []
            self._update_archive(pop.indivs[:1])
            best_fit = pop.indivs[0].fitness[0]
            self.best_individual = deepcopy(pop.indivs[0])
            epoch_best = best_fit
            stagnation, restarts, successes = 0, 0, 0
            shake_level = 1
            since_restart = None  # số thế hệ kể từ restart gần nhất (None = chưa restart)
            improved_in_epoch = False
            div = population_diversity(pop.indivs)
            history[0] = self._record(
                self.best_individual, pop.indivs[0], div, self.mutation_rate, 0, restarts, successes, shake_level
            )

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
                offspring.extend(new_immigrant(self.problem, self.heuristic_prob) for _ in range(n_imm))
                self._evaluate(offspring)
                pop.indivs.extend(offspring)
                short = pop.selection()
                if short > 0:
                    pop.indivs.extend(self._immigrants(short))
                    pop.indivs.sort(key=lambda ind: ind.fitness[0], reverse=True)

                # tiến độ của epoch hiện tại (quyết định đình trệ) và của toàn cục
                top = pop.indivs[0]
                if top.fitness[0] > epoch_best + 1e-9:
                    epoch_best = top.fitness[0]
                    stagnation = 0
                else:
                    stagnation += 1
                if top.fitness[0] > best_fit + 1e-9:
                    best_fit = top.fitness[0]
                    self.best_individual = deepcopy(top)
                    improved_in_epoch = True
                self._update_archive(pop.indivs[:1])

                event = ""
                if since_restart is not None:
                    since_restart += 1
                    if since_restart == self.isolation:
                        self._reinject(pop)
                        event = "  [REINJECT]"

                # điều kiện restart: đình trệ đủ lâu, hoặc quần thể đã hội tụ kiểu hình
                # (chỉ xét hội tụ khi đã qua giai đoạn cô lập: quần thể vừa đá quanh best vốn rất sát nhau)
                isolated = since_restart is not None and since_restart <= self.isolation
                if stagnation >= self.patience or (
                    not isolated and stagnation >= max(3, self.patience // 2) and self._converged(pop)
                ):
                    if restarts > 0:  # đánh giá epoch vừa kết thúc
                        if improved_in_epoch:
                            successes += 1
                            shake_level = 1
                        else:
                            shake_level += 1
                    self._update_archive(pop.indivs[:3])
                    self._restart(pop, shake_level)
                    event = "  [RESTART lv={}{}]".format(
                        shake_level, " HARD" if shake_level > self.max_shake else ""
                    )
                    if shake_level > self.max_shake:
                        shake_level = 1
                    restarts += 1
                    stagnation, since_restart, improved_in_epoch = 0, 0, False
                    epoch_best = pop.indivs[0].fitness[0]

                div = population_diversity(pop.indivs)
                history[gen + 1] = self._record(
                    self.best_individual, pop.indivs[0], div, pm, stagnation, restarts, successes, shake_level
                )
                if self.verbose:
                    print(
                        "Generation {}: best={:.2f} f1={} epoch={:.2f} div={:.3f} pm={:.2f}{}".format(
                            gen + 1,
                            self.best_individual.fitness[0],
                            self.best_individual.fitness[1],
                            pop.indivs[0].fitness[0],
                            div,
                            pm,
                            event,
                        )
                    )
        finally:
            if self.pool is not None:
                self.pool.close()
                self.pool.join()
                self.pool = None
        return history

    @staticmethod
    def _record(best, epoch_top, diversity, pm, stagnation, restarts, successes, shake_level):
        fitness, f1, f2, service_rate, violation, _ = best.fitness
        return {
            "epoch_fitness": epoch_top.fitness[0],
            "restart_success": successes,
            "shake_level": shake_level,
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
                        help="Thư mục dữ liệu gốc (mặc định ../data/data)")
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--max-gen", type=int, default=100)
    parser.add_argument("--crossover-rate", type=float, default=0.9)
    parser.add_argument("--mutation-rate", type=float, default=0.3)
    parser.add_argument("--mutation-rate-max", type=float, default=0.8)
    parser.add_argument("--tournament-size", type=int, default=2)
    parser.add_argument("--immigrant-rate", type=float, default=0.1)
    parser.add_argument("--seed-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=30,
                        help="Số thế hệ epoch hiện tại không cải thiện trước khi restart")
    parser.add_argument("--converge-threshold", type=float, default=0.002,
                        help="Restart sớm khi (best - median)/|best| < ngưỡng (hội tụ kiểu hình)")
    parser.add_argument("--archive-size", type=int, default=8,
                        help="Số cực trị cục bộ khác nhau lưu trong archive")
    parser.add_argument("--isolation", type=int, default=10,
                        help="Số thế hệ epoch mới chạy cô lập trước khi tái nhập archive")
    parser.add_argument("--max-shake", type=int, default=4,
                        help="Số mức leo thang cường độ đá; vượt quá -> restart cứng")
    parser.add_argument("--restart-fresh", type=float, default=0.3,
                        help="Tỉ lệ cá thể mới hoàn toàn khi restart (còn lại = archive bị đá)")
    parser.add_argument("--restart-strength", type=float, default=0.04,
                        help="Cường độ đá cho mỗi mức, tính theo tỉ lệ độ dài nhiễm sắc thể")
    parser.add_argument("--heuristic-prob", type=float, default=0.3,
                        help="Xác suất cá thể nhập cư được sinh bằng heuristic")
    parser.add_argument("--use-fill", action="store_true",
                        help="Dùng bước fill tham lam khi đánh giá trong lúc tiến hóa (chậm hơn, hội tụ nhanh hơn)")
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
                converge_threshold=args.converge_threshold,
                archive_size=args.archive_size,
                isolation=args.isolation,
                max_shake=args.max_shake,
                restart_fresh=args.restart_fresh,
                restart_strength=args.restart_strength,
                use_fill=args.use_fill,
                heuristic_prob=args.heuristic_prob,
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
                "num_restart_success": last["restart_success"],
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