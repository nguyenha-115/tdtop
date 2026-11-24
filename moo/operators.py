"""Toán tử tiến hóa trên chromosome random-key, đúng chữ ký mà moo_algorithm yêu cầu:

    create_solution(problem)                 -> np.ndarray      (Individual.gen_random)
    init_population(problem, pop_size)       -> list[Individual] (indi_list truyền cho run_*)
    crossover(problem, indi1, indi2)         -> (Individual, Individual)
    mutation(problem, indi)                  -> Individual
Các thuật toán DE / PSO (MODE, MOPSO, PFG-MOEA biến thể DE) tự làm phép toán trên chromosome rồi
Individual.repair() kẹp về [-pi, pi]; mã hóa random-key bảo đảm mọi vector như vậy đều giải mã được.
Số ngẫu nhiên lấy từ np.random toàn cục (các thuật toán cũng dùng np.random) để seed tái lập được.
"""
import math

import numpy as np

from config import (INIT_FILL, INIT_GREEDY, KEY_HIGH, KEY_LOW, KEY_MIN_ACTIVE, LS_RATE, LS_STEPS, MAX_STEP,
                    MUTATION_PROBS, PM_ETA, SBX_ETA, SBX_RATE)
from local_move import local_search, move_add, move_drop, move_route, move_swap
from population import Individual
from problem import Problem

RNG = np.random

# có thể chỉnh lúc chạy (utils.run_experiments gán SETTINGS["ls_rate"] = --ls-rate)
SETTINGS = {"ls_rate": LS_RATE, "ls_steps": LS_STEPS}


def _clip(x):
    return np.clip(x, KEY_LOW, KEY_HIGH)


def _new(chrom) -> Individual:
    return Individual(_clip(np.asarray(chrom, dtype=float)))


# ====================================================================
# Khởi tạo
# ====================================================================
def nn_order(P: Problem, nodes, start=0):
    """Sắp các nút đường theo láng giềng gần nhất (Manhattan) bắt đầu từ `start`."""
    nodes, cur, out = list(nodes), start, []
    while nodes:
        nxt = min(nodes, key=lambda x: P.truck_dist[cur][x])
        out.append(nxt)
        nodes.remove(nxt)
        cur = nxt
    return out


def sweep_routes(P: Problem, rng=RNG):
    """Chia nút đường cho K xe theo góc quét quanh kho (góc bắt đầu ngẫu nhiên), mỗi xe đi theo láng giềng gần nhất."""
    roads = list(P.road_ids)
    depot = P.nodes[0].coord
    off = rng.uniform(0, 2 * math.pi)
    roads.sort(key=lambda i: (math.atan2(P.nodes[i].coord[1] - depot[1],
                                         P.nodes[i].coord[0] - depot[0]) + off) % (2 * math.pi))
    K = P.num_trucks
    bounds = [0] + [round(len(roads) * (j + 1) / K) for j in range(K - 1)] + [len(roads)]
    return [nn_order(P, roads[bounds[j]:bounds[j + 1]]) for j in range(K)]


def make_chromosome(P: Problem, fill: float, greedy: bool, rng=RNG):
    """Chromosome với khoảng `fill` phần nút thông tin với tới được được bật.
    greedy: hành trình theo sweep + NN, bật các nút có giá trị/năng lượng cao (có nhiễu); ngược lại ngẫu nhiên."""
    x = np.empty(P.n_var)
    nr = P.num_road
    x[:nr] = P.routes_to_keys(sweep_routes(P, rng)) if greedy else rng.uniform(KEY_LOW, KEY_HIGH, nr)

    m = int(round(fill * len(P.reach_info)))
    ik = -rng.uniform(KEY_MIN_ACTIVE, KEY_HIGH, P.num_info)          # mặc định: tắt hết
    reach_j = [j for j, n in enumerate(P.info_ids) if P.reachable[n]]
    if m > 0 and reach_j:
        if greedy:
            sc = np.array([P.score[P.info_ids[j]] for j in reach_j]) * rng.uniform(0.7, 1.3, len(reach_j))
            order = [reach_j[i] for i in np.argsort(-sc)]
            on = order[:m]
            # nút đáng giá hơn có khóa lớn hơn -> được chèn trước
            vals = np.linspace(KEY_HIGH, KEY_MIN_ACTIVE, len(on)) if len(on) > 1 else np.array([KEY_HIGH])
            for j, v in zip(on, vals):
                ik[j] = v
        else:
            for j in rng.choice(reach_j, size=min(m, len(reach_j)), replace=False):
                ik[j] = rng.uniform(KEY_MIN_ACTIVE, KEY_HIGH)
    x[nr:] = ik
    return _clip(x)


def create_solution(problem: Problem):
    lo, hi = INIT_FILL
    return make_chromosome(problem, RNG.uniform(lo, hi), RNG.random() < INIT_GREEDY)


def init_population(problem: Problem, pop_size: int, greedy_ratio: float = INIT_GREEDY, fill=INIT_FILL):
    """Quần thể đầu: tỉ lệ bật nút trải đều trong [lo, hi] (phủ cả hai đầu mặt Pareto),
    `greedy_ratio` cá thể dùng heuristic."""
    lo, hi = fill
    pop = []
    for i in range(pop_size):
        q = lo + (hi - lo) * (i + RNG.random()) / pop_size
        pop.append(_new(make_chromosome(problem, q, RNG.random() < greedy_ratio)))
    RNG.shuffle(pop)
    return pop


# ====================================================================
# Lai ghép
# ====================================================================
def sbx(x1, x2, eta=SBX_ETA, prob_var=0.5, rng=RNG):
    """Simulated binary crossover có biên [KEY_LOW, KEY_HIGH]."""
    c1, c2 = x1.copy(), x2.copy()
    lo, hi = KEY_LOW, KEY_HIGH
    for i in range(len(x1)):
        if rng.random() > prob_var or abs(x1[i] - x2[i]) < 1e-14:
            continue
        y1, y2 = min(x1[i], x2[i]), max(x1[i], x2[i])
        r = rng.random()

        beta = 1.0 + 2.0 * (y1 - lo) / (y2 - y1)
        alpha = 2.0 - beta ** -(eta + 1)
        bq = (r * alpha) ** (1 / (eta + 1)) if r <= 1 / alpha else (1 / (2 - r * alpha)) ** (1 / (eta + 1))
        a = 0.5 * (y1 + y2 - bq * (y2 - y1))

        beta = 1.0 + 2.0 * (hi - y2) / (y2 - y1)
        alpha = 2.0 - beta ** -(eta + 1)
        bq = (r * alpha) ** (1 / (eta + 1)) if r <= 1 / alpha else (1 / (2 - r * alpha)) ** (1 / (eta + 1))
        b = 0.5 * (y1 + y2 + bq * (y2 - y1))

        a, b = min(max(a, lo), hi), min(max(b, lo), hi)
        if rng.random() < 0.5:
            a, b = b, a
        c1[i], c2[i] = a, b
    return c1, c2


def crossover(problem: Problem, indi1: Individual, indi2: Individual):
    """Uniform crossover (xác suất ngẫu nhiên mỗi lần), có thể giữ nguyên khối hành trình xe của một cha,
    sau đó SBX với xác suất SBX_RATE."""
    x1 = np.asarray(indi1.chromosome, dtype=float)
    x2 = np.asarray(indi2.chromosome, dtype=float)
    pa = RNG.uniform(0.2, 0.8)
    mask = RNG.random(len(x1)) < pa
    c1 = np.where(mask, x1, x2)
    c2 = np.where(mask, x2, x1)
    # khóa nút đường mang nghĩa theo cả khối (phân xe + thứ tự) -> 50% thừa hưởng nguyên khối
    nr = problem.num_road
    if nr and RNG.random() < 0.5:
        c1[:nr], c2[:nr] = x1[:nr], x2[:nr]
    if RNG.random() < SBX_RATE:
        c1, c2 = sbx(c1, c2)
    return _new(c1), _new(c2)


# ====================================================================
# Đột biến
# ====================================================================
def polynomial_mutation(x, eta=PM_ETA, prob_var=None, rng=RNG):
    x = x.copy()
    n = len(x)
    pv = prob_var if prob_var is not None else 1.0 / max(1, n)
    lo, hi = KEY_LOW, KEY_HIGH
    for i in range(n):
        if rng.random() >= pv:
            continue
        y = x[i]
        d1, d2 = (y - lo) / (hi - lo), (hi - y) / (hi - lo)
        r = rng.random()
        if r < 0.5:
            dq = (2 * r + (1 - 2 * r) * (1 - d1) ** (eta + 1)) ** (1 / (eta + 1)) - 1
        else:
            dq = 1 - (2 * (1 - r) + 2 * (r - 0.5) * (1 - d2) ** (eta + 1)) ** (1 / (eta + 1))
        x[i] = min(max(y + dq * (hi - lo), lo), hi)
    return x


def mutation(problem: Problem, indi: Individual, probs=MUTATION_PROBS, max_step=MAX_STEP):
    """Đột biến theo bài toán: bật / tắt / đổi nút thông tin, đổi hành trình xe, hoặc polynomial mutation.
    Sau đó tìm kiếm cục bộ với xác suất SETTINGS['ls_rate']."""
    P = problem
    x = np.array(indi.chromosome, dtype=float)
    r = RNG.random() * sum(probs.values())
    m = int(RNG.randint(1, max_step + 1))
    acc = 0.0
    for name, p in probs.items():
        acc += p
        if r < acc:
            break
    if name == "add":
        move_add(P, x, RNG, m)
    elif name == "drop":
        move_drop(P, x, RNG, m)
    elif name == "swap":
        for _ in range(m):
            move_swap(P, x, RNG)
    elif name == "route":
        move_route(P, x, RNG)
    else:
        x = polynomial_mutation(x)
    if SETTINGS["ls_rate"] > 0 and RNG.random() < SETTINGS["ls_rate"]:
        x, _ = local_search(P, x, RNG, SETTINGS["ls_steps"])
    return _new(x)
