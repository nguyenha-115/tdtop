"""
Cách chạy:
    python utils.py --algo nsga2 --filename S042_N10_R_R3.json --subfolder S/N10
    python utils.py --algo moead --subfolder S/N10 --max-gen 100 
    python utils.py --algo mode  --subfolder L/N100 --time-limit 120
"""
import importlib
import json
import random
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np

from config import EPS, INFEASIBLE_OFFSET, RESULT_ROOT, algo_params, base_parser
from local_move import decode
import operators
from problem import Problem, Route, Solution, Sortie, State, TruckPlan


# ====================================================================
# Fitness
# ====================================================================
def fitness_of_state(P: Problem, st: State):
    """(o1, o2) cực tiểu hóa: o1 = 1 - f1/F1_REF, o2 = f2/F2_REF; vi phạm T -> bị mọi lời giải khả thi trội."""
    v = st.violation
    if v > EPS:
        pen = 1.0 + INFEASIBLE_OFFSET + v
        return (pen, pen)
    o1, o2 = P.normalize(st.profit, st.energy)
    return (float(o1), float(o2))


def cal_fitness(problem: Problem, individual):
    """Hàm fitness truyền cho run_* (chạy được trong tiến trình con). Trả về tuple (o1, o2)."""
    return fitness_of_state(problem, decode(problem, individual.chromosome))


def is_feasible_fitness(objs):
    return objs[0] <= 1.0 + EPS and objs[1] <= 1.0 + 1e-6


# ====================================================================
# Giải mã -> Solution, kiểm tra ràng buộc, xuất JSON
# ====================================================================
def to_solution(st: State) -> Solution:
    P = st.P
    plans = []
    for k in range(P.num_trucks):
        nl = st.nl[k]
        arrival, departure = [0.0], [0.0]
        out = []
        for idx in range(len(nl) - 1):
            a, b = nl[idx], nl[idx + 1]
            t_launch = departure[-1]
            tg = st.tG[k][idx]
            for so in st.sorties[k][idx]:
                out.append(Sortie(truck=k, arc_index=idx, launch=a, land=b, nodes=list(so["seq"]),
                                  t_launch=t_launch, t_arrival=t_launch + so["F"],
                                  t_wait=max(0.0, tg - so["F"]), energy=so["E"]))
            arrival.append(t_launch + tg)
            departure.append(t_launch + st.dur[k][idx])
        used = len(nl) > 2 or bool(out)
        plans.append(TruckPlan(k, Route(list(nl), arrival, departure), out, used))
    return Solution(plans, st.violation)


def objectives(P: Problem, solution: Solution):
    """(f1, f2) gốc tính lại từ Solution."""
    return sum(P.profit[i] for i in solution.served), solution.energy


def verify_solution(P: Problem, solution: Solution):
    """Kiểm tra độc lập các ràng buộc MILP (38)-(60). Trả về danh sách lỗi (rỗng = hợp lệ)."""
    errs = []
    seen_road, seen_info = [], set()
    for plan in solution.plans:
        nl = plan.route.nodes
        seen_road += [n for n in nl if P.nodes[n].kind == "road"]
        if nl[0] != 0 or nl[-1] != P.end:
            errs.append("hành trình không bắt đầu/kết thúc ở kho (38)")
        by_arc = {}
        for s in plan.sorties:
            by_arc.setdefault(s.arc_index, []).append(s)
            if not s.nodes:
                errs.append("sortie rỗng (44)")
        if len(plan.sorties) > P.cap[plan.truck]:
            errs.append(f"xe {plan.truck + 1}: quá |Q_k| = {P.cap[plan.truck]} sortie")
        dep = 0.0
        for idx in range(len(nl) - 1):
            a, b = nl[idx], nl[idx + 1]
            tg = P.truck_time[a][b]
            lst = by_arc.get(idx, [])
            if len(lst) > P.R[plan.truck]:
                errs.append(f"xe {plan.truck + 1} cung {idx}: quá R_k = {P.R[plan.truck]} UAV (50)-(51)")
            Fmax = 0.0
            for s in lst:
                if s.launch != a or s.land != b:
                    errs.append("sortie không thu hồi tại điểm dừng kế tiếp (48)")
                path = [a] + s.nodes + [b]
                fly = sum(P.uav_time[x][y] for x, y in zip(path, path[1:]))
                srv = sum(P.service[i] for i in s.nodes)
                F = fly + srv
                E = (sum(P.uav_energy[x][y] for x, y in zip(path, path[1:])) + P.p_srv * srv
                     + P.p_hover * max(0.0, tg - F))
                if E > P.theta + 1e-6:
                    errs.append(f"sortie vượt pin (60): {E:.3f} > {P.theta:.3f}")
                if abs(E - s.energy) > 1e-5:
                    errs.append("năng lượng ghi nhận khác năng lượng tính lại (59)")
                if abs(s.t_launch - dep) > 1e-6:
                    errs.append("UAV không cất cánh lúc xe rời nút (54)")
                Fmax = max(Fmax, F)
                for i in s.nodes:
                    if P.nodes[i].kind != "info":
                        errs.append(f"nút {i} trong sortie không phải nút thông tin")
                    if i in seen_info:
                        errs.append(f"nút {i} bị phục vụ 2 lần (42)")
                    seen_info.add(i)
            dep += max(tg, Fmax)
        if dep > P.T + 1e-6 and solution.violation <= EPS:
            errs.append(f"vượt khung thời gian (58): {dep:.3f} > {P.T:.3f}")
    if sorted(seen_road) != sorted(P.road_ids):
        errs.append("nút đường không được thăm đúng 1 lần (39)-(40)")
    return errs


def solution_to_dict(P: Problem, solution: Solution, objs=None) -> Dict:
    """Tóm tắt lời giải (f1, f2, vi phạm, thống kê xe / UAV / sortie); không xuất chi tiết hành trình."""
    f1, f2 = objs if objs is not None else objectives(P, solution)
    oid = P.orig_id
    served = solution.served
    routes, total_sorties, used_uavs = [], 0, set()
    for plan in solution.plans:
        if not plan.used:
            continue
        k = plan.truck + 1
        R = P.R[plan.truck]
        free_at, trips = [0.0] * R, [0] * R
        uav_sorties = [[] for _ in range(R)]
        out = []
        for n, s in enumerate(sorted(plan.sorties, key=lambda s: (s.t_launch, s.arc_index)), start=1):
            free = [m for m in range(R) if free_at[m] <= s.t_launch + 1e-6]
            m = free[0] if free else min(range(R), key=lambda mm: free_at[mm])
            free_at[m] = s.t_arrival + s.t_wait
            trips[m] += 1
            sid, uid = f"T{k}-S{n}", f"T{k}-U{m + 1}"
            uav_sorties[m].append(sid)
            used_uavs.add(uid)
            out.append({
                "id": sid, "truck": k, "uav": uid, "uav_trip_no": trips[m],
                "launch": oid[s.launch], "land": oid[s.land],
                "path": [oid[s.launch]] + [oid[i] for i in s.nodes] + [oid[s.land]],
                "t_launch": s.t_launch, "t_arrival": s.t_arrival, "t_wait": s.t_wait,
                "energy_wh": s.energy, "served_nodes": [oid[i] for i in s.nodes],
            })
        total_sorties += len(out)
        routes.append({
            "id": k,
            "route": [oid[i] for i in plan.route.nodes],
            "arrival": plan.route.arrival,
            "departure": plan.route.departure,
            "uavs": [{"id": f"T{k}-U{m + 1}", "num_sorties": len(uav_sorties[m]), "sorties": uav_sorties[m]}
                     for m in range(R) if uav_sorties[m]],
            "sorties": out,
        })
    return {
        "f1_total_value": f1,
        "f2_total_energy_wh": f2,
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
    }


# ====================================================================
# Kho Pareto ngoài + hypervolume
# ====================================================================
def weakly_dominates(a, b):
    return a[0] <= b[0] + 1e-12 and a[1] <= b[1] + 1e-12


def hypervolume_2d(points, ref=(1.0, 1.0)) -> float:
    """HV của tập điểm cực tiểu hóa 2 chiều, điểm tham chiếu `ref` (mặc định (1,1) = f1 = 0, f2 = F2_REF)."""
    pts = sorted(p for p in points if p[0] < ref[0] and p[1] < ref[1])
    hv, best_y = 0.0, ref[1]
    for x, y in pts:
        if y < best_y:
            hv += (ref[0] - x) * (best_y - y)
            best_y = y
    return hv


class ParetoArchive:
    """Tập lời giải khả thi không bị trội (theo fitness chuẩn hóa), lưu kèm chromosome."""

    def __init__(self):
        self.items = []  # list[(objs tuple, chromosome)]

    def add(self, objs, x) -> bool:
        objs = (float(objs[0]), float(objs[1]))
        if not is_feasible_fitness(objs):
            return False
        for o, _ in self.items:
            if weakly_dominates(o, objs):
                return False
        self.items = [(o, c) for o, c in self.items if not weakly_dominates(objs, o)]
        self.items.append((objs, np.array(x, dtype=float)))
        return True

    @property
    def points(self):
        return [o for o, _ in self.items]

    def hypervolume(self):
        return hypervolume_2d(self.points)

    def __len__(self):
        return len(self.items)


class StopSearch(Exception):
    """Ném ra từ EvalTracker để dừng thuật toán (hết giờ / hội tụ)."""


class EvalTracker:
    """Thay multiprocessing.Pool trong module thuật toán: đánh giá (tuần tự hoặc song song), ghi kho Pareto và
    lịch sử HV theo thế hệ. Mỗi lần starmap = một thế hệ (thế hệ 0 = quần thể đầu)."""

    def __init__(self, problem: Problem, processes=1, time_limit=None, early_stop=None):
        self.P = problem
        self.processes = max(1, int(processes or 1))
        self.time_limit = time_limit
        self.early_stop = early_stop
        self.archive = ParetoArchive()
        self.history = {}
        self.gen = 0
        self.evals = 0
        self.best_hv, self.stall = -1.0, 0
        self.t0 = time.time()
        self._pool = None
        self._cache = {}
        self.stop_reason = None

    # giao diện giống multiprocessing (module thuật toán gọi multiprocessing.Pool(n))
    def as_module(self):
        return SimpleNamespace(Pool=lambda *a, **kw: _PoolHandle(self))

    def _real_pool(self):
        if self._pool is None:
            import multiprocessing
            self._pool = multiprocessing.Pool(self.processes)
        return self._pool

    def shutdown(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def starmap(self, func, args, chunksize=None):
        if self.stop_reason:
            raise StopSearch(self.stop_reason)
        if self.time_limit is not None and time.time() - self.t0 > self.time_limit:
            self.stop_reason = f"time limit {self.time_limit}s"
            raise StopSearch(self.stop_reason)
        args = list(args)
        if len(self._cache) > 200000:
            self._cache.clear()
        keys = [np.asarray(ind.chromosome, dtype=float).tobytes() for _, ind in args]
        todo = {}  # chromosome chưa từng đánh giá (bỏ trùng trong cùng thế hệ)
        for i, k in enumerate(keys):
            if k not in self._cache and k not in todo:
                todo[k] = i
        if todo:
            sub = [args[i] for i in todo.values()]
            if self.processes > 1 and len(sub) > 1:
                cs = chunksize or max(1, len(sub) // (4 * self.processes))
                res = self._real_pool().starmap(func, sub, chunksize=cs)
            else:
                res = [func(*a) for a in sub]
            for k, r in zip(todo, res):
                self._cache[k] = tuple(r)
            self.evals += len(todo)
        results = [self._cache[k] for k in keys]

        improved = False
        for (_, ind), r in zip(args, results):
            improved |= self.archive.add(r, ind.chromosome)
        self._record(improved)
        return results

    def _record(self, improved):
        hv = self.archive.hypervolume()
        pts = self.archive.points
        raw = [self.P.denormalize(*p) for p in pts]
        self.history[self.gen] = {
            "hypervolume": hv,
            "front_size": len(pts),
            "evaluations": self.evals,
            "f1_max": max((r[0] for r in raw), default=None),
            "f2_min": min((r[1] for r in raw), default=None),
            "time": time.time() - self.t0,
            "improved": bool(improved),
        }
        if hv > self.best_hv + 1e-12:
            self.best_hv, self.stall = hv, 0
        else:
            self.stall += 1
        if self.early_stop is not None and self.stall >= self.early_stop:
            self.stop_reason = f"HV không tăng sau {self.early_stop} thế hệ"
        self.gen += 1


class _PoolHandle:
    def __init__(self, tracker):
        self.tracker = tracker

    def starmap(self, func, iterable, chunksize=None):
        return self.tracker.starmap(func, iterable, chunksize)

    def close(self):
        pass

    def join(self):
        pass

    def terminate(self):
        pass


# ====================================================================
# Gọi thuật toán trong moo_algorithm
# ====================================================================
def _w2d(pop_size):
    from moead import init_weight_vectors_2d
    return init_weight_vectors_2d


# algo -> (module, hàm, danh sách tham số sau (processing_number, problem, indi_list, pop_size, max_gen))
ALGORITHMS = {
    "nsga2": ("nsga_ii", "run_nsga_ii", "ga"),
    "nsga3": ("nsga_iii", "run_nsga_iii", "ga"),
    "moead": ("moead", "run_moead", "moead"),
    "mode": ("mode", "run_mode", "de"),
    "mopso": ("mopso", "run_mopso", "pso"),
    "pfgmoea": ("pfg_moea", "run_pfgmoea", "pfg"),
    "pfgmoea_rand1": ("pfg_moea_knee", "run_pfgmoea_rand_1", "pfg_de"),
    "pfgmoea_best1": ("pfg_moea_knee", "run_pfgmoea_best_1", "pfg_de"),
    "pfgmoea_best1_pareto": ("pfg_moea_knee", "run_pfgmoea_best_1_pareto", "pfg_de"),
    "pfgmoea_ctb1": ("pfg_moea_knee", "run_pfgmoea_current_to_best_1", "pfg_de"),
    "pfgmoea_randbest1": ("pfg_moea_knee", "run_pfgmoea_rand_best_1", "pfg_de"),
}


def run_algorithm(algo: str, problem: Problem, params: Dict, processes=1, time_limit=None, early_stop=None):
    """Chạy một thuật toán của moo_algorithm trên `problem`. Trả về (tracker, algo_history, stop_reason)."""
    mod_name, fn_name, kind = ALGORITHMS[algo]
    mod = importlib.import_module(mod_name)
    tracker = EvalTracker(problem, processes, time_limit, early_stop)
    old_mp = mod.multiprocessing
    mod.multiprocessing = tracker.as_module()
    pop_size, max_gen = int(params["pop_size"]), int(params["max_gen"])
    indi_list = operators.init_population(problem, pop_size)
    head = (processes, problem, indi_list, pop_size, max_gen)
    xo, mu, fit = operators.crossover, operators.mutation, cal_fitness
    if kind == "ga":
        call = head + (xo, mu, params["crossover_rate"], params["mutation_rate"], fit)
    elif kind == "moead":
        call = head + (int(params["neighbor_size"]), _w2d(pop_size), xo, mu, fit)
    elif kind == "de":
        call = head + (params["F"], params["CR"], fit)
    elif kind == "pso":
        call = head + (params["w"], params["c1"], params["c2"], fit)
    elif kind == "pfg":
        call = head + (int(params["GK"]), params["sigma"], xo, mu, params["crossover_rate"],
                       params["mutation_rate"], fit)
    else:  # pfg_de
        call = head + (int(params["GK"]), params["sigma"], params["F"], params["CR"], fit)

    algo_history, reason = None, None
    try:
        algo_history = getattr(mod, fn_name)(*call)
    except StopSearch as e:
        reason = str(e)
    finally:
        mod.multiprocessing = old_mp
        tracker.shutdown()
    return tracker, algo_history, reason


# ====================================================================
# Vào / ra
# ====================================================================
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
        for m in matches:
            files.append((m, m.parent.relative_to(data_root).as_posix()))
    elif subfolder:
        folder = data_root / subfolder
        if not folder.is_dir():
            print(f"Không tìm thấy thư mục {folder}")
            return []
        files = [(f, subfolder) for f in sorted(folder.glob("*.json"))]
    else:
        print("Cần truyền --filename và/hoặc --subfolder")
    return files


def save_result(result: Dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved result to {output_path}")


def front_to_list(P: Problem, archive: ParetoArchive):
    out, all_errs = [], []
    for objs, x in sorted(archive.items, key=lambda t: (-t[0][0], t[0][1])):
        st = decode(P, x)
        sol = to_solution(st)
        errs = verify_solution(P, sol)
        if any(abs(a - b) > 1e-9 for a, b in zip(fitness_of_state(P, st), objs)):
            errs.append("giải mã lại cho mục tiêu khác fitness đã lưu")
        all_errs.extend(errs)
        f = objectives(P, sol)
        out.append({
            "f1_total_value": f[0],
            "f2_total_energy_wh": f[1],
            "objectives_normalized": list(objs),
            "violation_h": sol.violation,
            "verify_errors": errs,
            "chromosome": [round(float(v), 6) for v in x],
            "solution": solution_to_dict(P, sol, f),
        })
    return out, all_errs


def build_result(P: Problem, tracker: EvalTracker, elapsed: float, params: Dict = None, stop_reason=None) -> Dict:
    """JSON kết quả: chỉ số mặt Pareto + toàn bộ lời giải không trội (định dạng dùng bởi compare.py, visualize.py)."""
    pareto, errs = front_to_list(P, tracker.archive)
    f1s = [p["f1_total_value"] for p in pareto]
    f2s = [p["f2_total_energy_wh"] for p in pareto]
    history = tracker.history
    return {
        "time": elapsed,
        "stop_reason": stop_reason,
        "evaluations": tracker.evals,
        "history": history,
        "hypervolume": tracker.archive.hypervolume(),
        "hv_reference": {"f1": 0.0, "f2_wh": P.f2_ref, "f1_norm": P.f1_ref},
        "pareto_size": len(pareto),
        "ideal_point": {"f1": max(f1s), "f2_wh": min(f2s)} if pareto else None,
        "nadir_point": {"f1": min(f1s), "f2_wh": max(f2s)} if pareto else None,
        "improved_generations": sum(1 for g in history.values() if g.get("improved")),
        "num_generations": max(history.keys()) if history else 0,
        "verify_errors": sorted(set(errs)),
        "params": params or {},
        "pareto_front": pareto,
    }


def run_experiments(args):
    """Chạy args.algo trên các file được chọn, lưu vào moo/result/<algo>/<subfolder>/<file>.json."""
    files = collect_files(Path(args.data_root), args.filename, args.subfolder)
    if not files:
        return
    params = algo_params(args)
    operators.SETTINGS["ls_rate"] = args.ls_rate
    print(f"\n{'=' * 80}\n{args.algo.upper()} (TD-TOP, max f1 / min f2)  params={params}\n{'=' * 80}")
    print(f"Found {len(files)} file(s)")
    run_params = dict(params, seed=args.seed, processes=args.processes, s_max=args.s_max,
                      time_limit=args.time_limit, early_stop=args.early_stop, ls_rate=args.ls_rate)

    for data_file, sub in files:
        print(f"\n{'-' * 80}\nProcessing: {sub}/{data_file.name}\n{'-' * 80}")
        try:
            random.seed(args.seed)
            np.random.seed(args.seed)
            problem = Problem(str(data_file), s_max_default=args.s_max)
            t0 = time.time()
            tracker, _, reason = run_algorithm(args.algo, problem, params, args.processes,
                                               args.time_limit, args.early_stop)
            elapsed = time.time() - t0
            if reason:
                print(f"  Dừng sớm: {reason}")
            result = build_result(problem, tracker, elapsed, run_params, reason)
            if result["verify_errors"]:
                print("  CẢNH BÁO: lời giải vi phạm ràng buộc:", result["verify_errors"])
            save_result(result, RESULT_ROOT / args.algo / sub / data_file.name)
            print(f"  Completed in {elapsed:.2f}s, {tracker.evals} evaluations, {tracker.gen} generations")
            print(f"  Pareto front: {result['pareto_size']} solutions, HV={result['hypervolume']:.5f}")
            for p in result["pareto_front"]:
                print(f"    f1={p['f1_total_value']:>7}  f2={p['f2_total_energy_wh']:.3f} Wh")
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 80}\n{args.algo.upper()} experiments completed!\n{'=' * 80}\n")


def main(argv=None, algo=None):
    """algo: thuật toán mặc định (các script nsga2.py, moead.py, ... truyền vào); vẫn đổi được bằng --algo."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = base_parser(f"{algo.upper()} cho TD-TOP đa mục tiêu" if algo else None)
    if algo:
        parser.set_defaults(algo=algo)
    run_experiments(parser.parse_args(argv))


if __name__ == "__main__":
    main()
