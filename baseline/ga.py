import argparse
import random
import time
from pathlib import Path
from typing import Dict, Any

from problem import Problem
from utils import (
    prepare,
    State,
    construct,
    random_construct,
    random_routes,
    lns,
    crossover,
    mutate,
    route_neighbors,
    population_diversity,
    decode,
    cal_objective,
    verify_solution,
    solution_to_dict,
    save_result,
)

EPS = 1e-9


class GASolver:
    def __init__(
        self,
        problem: Problem,
        mode: str = "convergence",
        ramp: float = 0.5,
        ramp_power: float = 3.0,
        pop_size: int = 20,
        max_gen: int = 100,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.3,
        tournament_size: int = 2,
        init_iters: int = 150,
        ls_iters: int = None,
        best_iters: int = 150,
        intens_tries: int = 4,
        max_per_route: int = None,
        route_every: int = 1,
        sweep_base: int = 3,
        sweep_limit: int = 60,
        sweep_iters: int = 60,
        deepen_routes: int = 3,
        deep_iters: int = 300,
        tau0: float = 4.0,
        tau1: float = 0.3,
        early_stop: int = None,
        time_limit: float = None,
        processing_number: int = 1,  # giữ tham số cho tương thích; LNS chạy tuần tự trong tiến trình chính
        seed: int = 42,
        verbose: bool = True,
    ):
        assert mode in ("pure", "convergence", "performance")
        self.mode = mode
        self.P = problem
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.init_iters = init_iters
        self.ls_iters = ls_iters if ls_iters is not None else (0 if mode == "pure" else 60)
        self.best_iters = best_iters
        self.intens_tries = max(1, intens_tries)
        self.pure = mode == "pure"            # GA thuần: không LNS, không khai thác tinh hoa
        self.cold = mode in ("pure", "convergence")  # khởi tạo ngẫu nhiên "khởi đầu lạnh"
        self.ramp_power = max(1.0, ramp_power)  # >1: cường độ tăng chậm ở đầu, nhanh ở cuối (r**p)
        self.ramp = max(1e-6, ramp)           # tỉ lệ số thế hệ dùng để tăng dần cường độ tối ưu cục bộ
        self._light = self.pure
        self._ls = 0
        if max_per_route:
            self.max_per_route = max_per_route
        else:  # chế độ hội tụ: không "ngăn" theo hành trình (µ+λ thuần túy)
            self.max_per_route = pop_size if self.pure else max(2, pop_size // 5)
        self.route_every = max(1, route_every)
        self.sweep_base = max(1, sweep_base)
        self.sweep_limit = sweep_limit
        self.sweep_iters = sweep_iters
        self.deepen_routes = max(1, deepen_routes)
        self.deep_iters = deep_iters
        self.archive = {}  # hành trình -> lời giải tốt nhất từng gặp trên hành trình đó
        self.tau0 = tau0
        self.tau1 = tau1
        self.early_stop = early_stop
        self.time_limit = time_limit
        self.seed = seed
        self.verbose = verbose
        self.rng = random.Random(seed)
        self.best_individual = None
        self.deadline = None

    # ------------------------------------------------------------------ tiện ích
    def _timeout(self):
        return bool(self.deadline) and time.time() > self.deadline

    def _new_individual(self, idx):
        P, rng = self.P, self.rng
        if self.cold:
            return random_construct(P, random_routes(P, rng, heuristic=False), rng)
        routes = random_routes(P, rng, heuristic=(idx == 0))
        st = construct(P, routes, rng)
        return lns(st, self.init_iters, rng, self.tau0, self.tau1, deadline=self.deadline)

    def _tournament(self, pop):
        cands = self.rng.sample(pop, min(self.tournament_size, len(pop)))
        return max(cands, key=lambda s: s.score)

    @staticmethod
    def _rkey(st):
        return tuple(tuple(r) for r in st.routes)

    def _archive_add(self, st):
        key = self._rkey(st)
        cur = self.archive.get(key)
        if cur is None or st.score > cur.score + EPS:
            self.archive[key] = st

    def _select(self, pop, size):
        """(µ+λ) có "ngăn" theo hành trình: loại bản sao (cùng hành trình + cùng tập nút phục vụ) và mỗi hành trình
        chỉ giữ tối đa `max_per_route` cá thể, nên quần thể không bị một hành trình chiếm hết. Cá thể tốt nhất luôn được giữ.
        Thiếu chỗ thì mới lấy bù từ phần bị loại do vượt hạn mức."""
        pop.sort(key=lambda s: s.score, reverse=True)
        seen, per, uniq, spill = set(), {}, [], []
        for st in pop:
            rk = self._rkey(st)
            ck = (rk, frozenset(st.served))
            if ck in seen:
                continue
            seen.add(ck)
            if per.get(rk, 0) >= self.max_per_route:
                spill.append(st)
                continue
            per[rk] = per.get(rk, 0) + 1
            uniq.append(st)
            if len(uniq) == size:
                break
        for st in spill:
            if len(uniq) >= size:
                break
            uniq.append(st)
        uniq.sort(key=lambda s: s.score, reverse=True)
        return uniq

    # ------------------------------------------------------------------ khai thác
    def _intensify(self, pop, stagnation, scale=1.0):
        """LNS riêng cho tinh hoa: mỗi lần thử tăng cường độ ruin; dừng khi vượt kỷ lục. True nếu cải thiện."""
        P, rng = self.P, self.rng
        n_it = int(self.best_iters * scale)
        if n_it < 3:
            return False
        base = pop[0]
        for t in range(self.intens_tries):
            if self._timeout():
                break
            src = base if (t % 2 == 0 or len(pop) < 2) else rng.choice(pop[: max(2, len(pop) // 4)])
            strength = 1.0 + 0.5 * t + 0.25 * min(stagnation, 6)
            cand = lns(src, n_it, rng, 1.5, self.tau1, strength, self.deadline)
            if cand.signature() not in {s.signature() for s in pop}:
                pop.append(cand)
            if cand.score > base.score + EPS:
                return True
        return False

    def _route_phase(self, pop, scale=1.0):
        """Đánh giá hai tầng các hành trình (chống nhiễu của LNS ngắn):
        (1) quét láng giềng của vài hành trình tốt nhất trong kho, mỗi hành trình MỚI chỉ cần LNS ngắn (kết quả được lưu
            nên không đánh giá lại);
        (2) đào sâu: các hành trình đứng đầu kho được LNS dài hơn (cộng dồn qua các thế hệ), nên tiềm năng thật của
            hành trình lộ ra dần thay vì chờ may mắn mới thấy."""
        P, rng = self.P, self.rng
        lim = int(self.sweep_limit * scale)
        it_short = int(self.sweep_iters * scale)
        it_deep = int(self.deep_iters * scale)
        top = sorted(self.archive.values(), key=lambda s: -s.score)[: self.sweep_base] if lim >= 1 else []
        for base in top:
            for routes in route_neighbors(P, base.routes, lim, rng):
                if self._timeout():
                    return
                if tuple(tuple(r) for r in routes) in self.archive:
                    continue
                ns = base.with_routes(routes, rng)
                ns.greedy_fill(rng, 0.0)
                while ns.has_free_slot() and ns.build_sortie(rng, 3):
                    ns.greedy_fill(rng, 0.0, only_touched=True)
                ns.polish(rng)
                ns.clear_touched()
                ns = lns(ns, it_short, rng, 2.0, self.tau1, 1.0, self.deadline)
                self._archive_add(ns)
                pop.append(ns)
        if it_deep < 3:
            return
        for base in sorted(self.archive.values(), key=lambda s: -s.score)[: self.deepen_routes]:
            if self._timeout():
                return
            cand = lns(base, it_deep, rng, 2.0, self.tau1, 1.0, self.deadline)
            self._archive_add(cand)
            pop.append(cand)

    # ------------------------------------------------------------------ vòng lặp chính
    def _progress(self, gen):
        """Tiến độ r in [0,1]: cường độ tối ưu cục bộ (LNS mỗi con, khai thác tinh hoa, quét/đào sâu hành trình) tăng
        tuyến tính theo thế hệ trong `ramp` x max_gen thế hệ đầu rồi giữ nguyên. Chỉ dùng ở chế độ convergence."""
        if self.mode != "convergence":
            return 1.0
        return min(1.0, gen / max(1.0, self.ramp * self.max_gen))

    def run(self) -> Dict[str, Any]:
        P, rng = self.P, self.rng
        prepare(P)
        start = time.time()
        self.deadline = start + self.time_limit if self.time_limit else None
        history = {}

        if self.verbose:
            print("Population initialization")
        pop = [self._new_individual(i) for i in range(self.pop_size)]
        for st in pop:
            self._archive_add(st)
        pop = self._select(pop, self.pop_size)
        self.best_individual = pop[0].copy()
        best_fit = pop[0].score
        stagnation = 0
        history[0] = self._record(pop[0], population_diversity(pop, rng), 0, 0, True, len({self._rkey(x) for x in pop}), pop)

        for gen in range(self.max_gen):
            if self._timeout():
                if self.verbose:
                    print("Time limit reached")
                break

            r = self._progress(gen)
            s = r ** self.ramp_power  # cường độ tối ưu cục bộ in [0,1], tăng liên tục (không có công tắc bật/tắt cứng)
            ls_f = 0.0 if self.pure else self.ls_iters * s  # số vòng LNS/con kỳ vọng (có thể < 1)

            # --- 1. sinh con ---
            offspring = []
            n_imm = 0
            while len(offspring) < self.pop_size and not self._timeout():
                p1 = self._tournament(pop)
                p2 = self._tournament(pop)
                for _ in range(5):
                    if p2 is not p1:
                        break
                    p2 = self._tournament(pop)
                # mỗi con dùng toán tử "mạnh" (chèn tham lam + tinh chỉnh) với xác suất s, ngược lại dùng toán tử ngẫu nhiên thuần
                self._light = self.pure or rng.random() >= s
                if rng.random() < self.crossover_rate:
                    child = crossover(P, p1, p2, rng, light=self._light)
                else:
                    child = p1.copy()
                if rng.random() < self.mutation_rate:
                    child = mutate(P, child, rng, 1.0 + min(2.0, stagnation / 3.0), light=self._light)
                # phần thập phân của ls_f là xác suất con được thêm một vòng -> cường độ tăng liên tục, không có bậc thang
                n_ls = int(ls_f) + (1 if rng.random() < ls_f - int(ls_f) else 0)
                child = lns(child, n_ls, rng, self.tau0, self.tau1, deadline=self.deadline)
                self._archive_add(child)
                offspring.append(child)

            # --- 2. chọn lọc (µ+λ) + nhập cư nếu thiếu ---
            pop = self._select(pop + offspring, self.pop_size)
            tries = 0
            while len(pop) < self.pop_size and tries < 5 and not self._timeout():
                new = [self._new_individual(1 + rng.randrange(1000)) for _ in range(self.pop_size - len(pop))]
                for st in new:
                    self._archive_add(st)
                pop.extend(new)
                n_imm += len(new)
                pop = self._select(pop, self.pop_size)
                tries += 1

            # --- 3. khai thác tinh hoa; đình trệ thì quét hành trình ---
            if not self.pure:
                if (gen + 1) % self.route_every == 0:
                    self._route_phase(pop, s)
                    pop = self._select(pop, self.pop_size)
                improved_here = pop[0].score > best_fit + EPS
                if not improved_here:
                    self._intensify(pop, stagnation, s)
                    pop = self._select(pop, self.pop_size)

            top = pop[0]
            improved = top.score > best_fit + EPS
            if improved:
                best_fit = top.score
                self.best_individual = top.copy()
                stagnation = 0
            else:
                stagnation += 1
            div = population_diversity(pop, rng)
            history[gen + 1] = self._record(self.best_individual, div, stagnation, n_imm, improved, len({self._rkey(x) for x in pop}), pop)
            if self.verbose:
                print(
                    "Generation {}: best={:.2f} f1={} served={} div={:.3f} routes={} archive={} t={:.0f}s{}".format(
                        gen + 1, self.best_individual.score, self.best_individual.profit,
                        len(self.best_individual.served), div, len({self._rkey(x) for x in pop}), len(self.archive),
                        time.time() - start, "  *" if improved else "",
                    ),
                    flush=True,
                )
            if self.early_stop and stagnation >= self.early_stop:
                if self.verbose:
                    print("Early stop")
                break
        return history

    def _record(self, best: State, diversity, stagnation, n_imm, improved, distinct_routes=0, pop=None):
        fitness, f1, f2, service_rate, violation, _ = best.fitness
        return {
            "fitness": fitness,
            "profit": f1,
            "energy": f2,
            "service_rate": service_rate,
            "violation": violation,
            "diversity": diversity,
            "mutation_rate": self.mutation_rate,
            "immigrants": n_imm,
            "stagnation": stagnation,
            "improved": improved,
            "distinct_routes": distinct_routes,
            "avg_profit": (sum(x.profit for x in pop) / len(pop)) if pop else None,
            "worst_profit": min(x.profit for x in pop) if pop else None,
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
        python ga.py --subfolder L/N100
    """
    parser = argparse.ArgumentParser(description="Run GA solver for TD-TOP")
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--subfolder", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="../data/data")
    parser.add_argument("--mode", choices=["convergence", "pure", "performance"], default="convergence",
                        help="convergence: khởi đầu lạnh + tăng dần cường độ LNS -> đồ thị hội tụ chuẩn VÀ nghiệm cuối tốt; "
                             "pure: GA thuần (không LNS) -> đồ thị chuẩn nhưng nghiệm kém hơn; "
                             "performance: memetic GA đủ mạnh ngay từ đầu -> nghiệm tốt nhất, hội tụ rất nhanh")
    parser.add_argument("--ramp", type=float, default=0.5, help="(convergence) tỉ lệ thế hệ dùng để tăng dần cường độ LNS")
    parser.add_argument("--ramp-power", type=float, default=3.0, help="(convergence) số mũ của lộ trình tăng cường độ (r**p); lớn hơn = đoạn đầu dài hơn")
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--max-gen", type=int, default=100)
    parser.add_argument("--crossover-rate", type=float, default=0.9)
    parser.add_argument("--mutation-rate", type=float, default=0.3)
    parser.add_argument("--tournament-size", type=int, default=2)
    parser.add_argument("--init-iters", type=int, default=150, help="Số vòng LNS cho cá thể khởi tạo")
    parser.add_argument("--ls-iters", type=int, default=None,
                        help="Số vòng LNS (Lamarck) cho mỗi con. Mặc định: 0 (convergence) / 60 (performance)")
    parser.add_argument("--best-iters", type=int, default=150, help="Số vòng LNS mỗi lần khai thác tinh hoa")
    parser.add_argument("--intens-tries", type=int, default=4, help="Số lần khai thác tinh hoa tối đa mỗi thế hệ")
    parser.add_argument("--max-per-route", type=int, default=None, help="Số cá thể tối đa mỗi hành trình trong quần thể")
    parser.add_argument("--route-every", type=int, default=1, help="Chạy giai đoạn quét/đào sâu hành trình mỗi N thế hệ")
    parser.add_argument("--sweep-base", type=int, default=3, help="Số hành trình tốt nhất dùng làm gốc để quét láng giềng")
    parser.add_argument("--sweep-limit", type=int, default=60)
    parser.add_argument("--sweep-iters", type=int, default=60, help="Số vòng LNS ngắn cho hành trình mới")
    parser.add_argument("--deepen-routes", type=int, default=3, help="Số hành trình đầu kho được LNS sâu mỗi thế hệ")
    parser.add_argument("--deep-iters", type=int, default=300, help="Số vòng LNS sâu mỗi lần")
    parser.add_argument("--early-stop", type=int, default=None)
    parser.add_argument("--time-limit", type=float, default=None, help="Giới hạn thời gian mỗi instance (giây)")
    parser.add_argument("--s-max", type=int, default=2, help="Số sortie tối đa/UAV nếu dữ liệu không khai báo")
    parser.add_argument("--processing-number", type=int, default=1)
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
                mode=args.mode,
                ramp=args.ramp,
                ramp_power=args.ramp_power,
                pop_size=args.pop_size,
                max_gen=args.max_gen,
                crossover_rate=args.crossover_rate,
                mutation_rate=args.mutation_rate,
                tournament_size=args.tournament_size,
                init_iters=args.init_iters,
                ls_iters=args.ls_iters,
                best_iters=args.best_iters,
                intens_tries=args.intens_tries,
                max_per_route=args.max_per_route,
                route_every=args.route_every,
                sweep_base=args.sweep_base,
                sweep_limit=args.sweep_limit,
                sweep_iters=args.sweep_iters,
                deepen_routes=args.deepen_routes,
                deep_iters=args.deep_iters,
                early_stop=args.early_stop,
                time_limit=args.time_limit,
                processing_number=args.processing_number,
                seed=args.seed,
            )
            history = solver.run()
            end_time = time.time()

            best = solver.best_individual
            solution = decode(best)
            errs = verify_solution(problem, solution)
            if errs:
                print("  CẢNH BÁO: lời giải vi phạm ràng buộc:", errs)
            fit = cal_objective(problem, solution)

            result = {
                "time": end_time - start_time,
                "history": history,
                "best_fitness": fit[0],
                "best_profit": fit[1],
                "best_energy": fit[2],
                "best_service_rate": len(solution.served) / max(1, problem.num_info),
                "improved_generations": sum(1 for g in history.values() if g["improved"]),
                "num_generations": max(history.keys()),
                "verify_errors": errs,
                "best_solution": solution_to_dict(problem, solution, fit),
            }
            save_result(result, result_dir / sub / data_file.name)
            print(f"  Completed in {result['time']:.2f} seconds")
            print(f"  Best profit f1: {result['best_profit']:.2f}  (improved in {result['improved_generations']}/{result['num_generations']} gens)")

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\n{'=' * 80}\nGA experiments completed!\n{'=' * 80}\n")


if __name__ == "__main__":
    main()