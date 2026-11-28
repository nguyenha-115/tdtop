"""Toán tử biến đổi của KL-NSGA-II-RL (Mục 6.3).

    O1  lấy mẫu chính sách pi_theta(. | I, lambda)                      - khám phá toàn cục
    O2  phá-xây có hướng dẫn: giữ Pha A, xóa sortie trên rho cung, giải mã lại Pha B
        (teacher forcing cho phần giữ, lấy mẫu từ pi_theta cho phần xóa)  - khai thác cục bộ có học
    O3  lai ghép lộ trình: OX trên giant tour -> tách theo xe (sửa nếu vi phạm T) -> Pha B bằng pi_theta,
        giữ sortie của các cung còn tồn tại                              - kết hợp cấu trúc xe
    O4  tìm kiếm cục bộ trên sortie (chèn / xóa / chuyển / hoán đổi / 2-opt) kiểm khả thi bằng simulator
    O5  thang Pareto (Pareto ladder): đánh bóng cha rồi đi dọc mặt Pareto bằng chuỗi chèn tham lam
        (tăng f1 với chi phí năng lượng biên nhỏ nhất) và chuỗi xóa tham lam (giảm f2 với tổn thất f1 nhỏ nhất);
        mọi bước trung gian đều là con -> lấp các khoảng trống của mặt Pareto chỉ trong một lần gọi

polish(st): cải thiện tất định một lời giải mà không đổi tập nút phục vụ - tối ưu thứ tự bay từng sortie
    (2-opt / or-opt) và chuyển sortie sang cung rẻ hơn, rồi chèn lại nút nếu còn chỗ miễn phí năng lượng.
    Áp cho con của O1-O3 (tham số polish) -> mạng chỉ cần chọn đúng tập nút / cấu trúc, phần tinh chỉnh
    năng lượng do đánh bóng đảm nhận (và con đã đánh bóng được đưa lại cho mạng học bắt chước).

Mọi hàm nhận `ctx` (Ctx) và trả về danh sách Sol.
"""

import numpy as np

import local_move as lm  # moo/local_move.py (moo/ đã nằm trong sys.path qua .utils)

from .environment import Env, run_batch
from .utils import Sol, build_state, route_time, scalarize, sortie_metrics

OPS = ("O1", "O2", "O3", "O4", "O5")


class Ctx:
    """Ngữ cảnh một instance: policy, encoder cache, RNG, tập lambda."""

    def __init__(self, inst, policy, enc, rng, Lam, n_aug=8, scalar="tcheby"):
        self.inst, self.policy, self.enc, self.rng, self.Lam = inst, policy, enc, rng, Lam
        self.n_aug, self.scalar = n_aug, scalar
        self._aug = 0

    def next_aug(self):
        self._aug = (self._aug + 1) % self.n_aug
        return self._aug

    def decode(self, specs, greedy=False):
        """specs: list dict(lam, route_guide, arc_guide, guide_free, blocked). Trả (list Sol, list Env)."""
        envs = [Env(self.inst, s["lam"], s.get("route_guide"), s.get("arc_guide"), s.get("guide_free", True),
                    s.get("blocked"), aug=self.next_aug()) for s in specs]
        run_batch(self.policy, self.enc, envs, greedy=greedy)
        return [e.solution() for e in envs], envs

    def random_lam(self):
        return self.Lam[self.rng.integers(len(self.Lam))]


# --------------------------------------------------------------------- O1
def o1_sample(ctx, lams, greedy=False):
    sols, _ = ctx.decode([{"lam": l} for l in lams], greedy=greedy)
    return sols


# --------------------------------------------------------------------- O2
def all_arcs(routes):
    return [(k, idx) for k, r in enumerate(routes) for idx in range(len(r) + 1)]


def o2_destroy_rebuild(ctx, parents, lams, rho):
    specs = []
    for par, lam in zip(parents, lams):
        arcs = all_arcs(par.routes)
        n_del = min(len(arcs), max(1, int(round(rho * len(arcs)))))
        gone = {arcs[i] for i in ctx.rng.choice(len(arcs), n_del, replace=False)}
        guide, blocked = {}, []
        for key in arcs:
            if key in gone:
                continue
            guide[key] = par.arcs.get(key, [])
            for s in guide[key]:
                blocked += s
        specs.append({"lam": lam, "route_guide": par.routes, "arc_guide": guide, "guide_free": True,
                      "blocked": blocked})
    sols, _ = ctx.decode(specs)
    return sols


# --------------------------------------------------------------------- O3
def _giant(routes):
    seq = []
    for k, r in enumerate(routes):
        seq += r
        if k < len(routes) - 1:
            seq.append(-(k + 1))
    return seq


def _ox(a, b, rng):
    L = len(a)
    if L < 2:
        return list(a)
    i, j = sorted(rng.choice(L + 1, 2, replace=False))
    if i == j:
        j = min(L, j + 1)
    child = [None] * L
    child[i:j] = a[i:j]
    used = set(a[i:j])
    fill = [x for x in b if x not in used]
    it = iter(fill)
    for p in list(range(0, i)) + list(range(j, L)):
        child[p] = next(it)
    return child


def _split(seq, K):
    routes = [[]]
    for x in seq:
        if x < 0:
            routes.append([])
        else:
            routes[-1].append(x)
    while len(routes) < K:
        routes.append([])
    return routes


def repair_routes(inst, routes, iters=60):
    """Chuyển nút từ xe quá T sang xe khác (chèn rẻ nhất). Trả None nếu không sửa được."""
    routes = [list(r) for r in routes]
    tG, end, T = inst.tG, inst.end, inst.T
    for _ in range(iters):
        times = [route_time(inst, r) for r in routes]
        bad = [k for k in range(len(routes)) if times[k] > T + 1e-9]
        if not bad:
            return routes
        if len(routes) == 1:
            return None
        k = max(bad, key=lambda q: times[q])
        r = routes[k]
        nl = [0] + r + [end]
        gains = [tG[nl[i], nl[i + 1]] + tG[nl[i + 1], nl[i + 2]] - tG[nl[i], nl[i + 2]] for i in range(len(r))]
        u = r.pop(int(np.argmax(gains)))
        best = None
        for q in range(len(routes)):
            if q == k:
                continue
            nq = [0] + routes[q] + [end]
            for p in range(len(nq) - 1):
                nt = times[q] + tG[nq[p], u] + tG[u, nq[p + 1]] - tG[nq[p], nq[p + 1]]
                if best is None or nt < best[0]:
                    best = (nt, q, p)
        routes[best[1]].insert(best[2], u)
    return None


def o3_route_crossover(ctx, pairs, lams):
    inst = ctx.inst
    K = inst.K
    specs = []
    for (p1, p2), lam in zip(pairs, lams):
        child = repair_routes(inst, _split(_ox(_giant(p1.routes), _giant(p2.routes), ctx.rng), K))
        if child is None:
            child = [list(r) for r in p1.routes]
        cmap = {}
        for k, r in enumerate(child):
            nl = [0] + r + [inst.end]
            for idx in range(len(nl) - 1):
                cmap.setdefault((nl[idx], nl[idx + 1]), []).append((k, idx))
        guide, used, cnt = {}, set(), [0] * K
        for par in (p1, p2):
            for (k, idx), seqs in par.arcs.items():
                nl = [0] + par.routes[k] + [inst.end]
                tgt = [t for t in cmap.get((nl[idx], nl[idx + 1]), []) if t not in guide]
                if not tgt:
                    continue
                k2, i2 = tgt[0]
                keep = []
                for s in seqs:
                    if len(keep) >= inst.R[k2] or cnt[k2] >= inst.cap[k2] or any(x in used for x in s):
                        continue
                    keep.append(list(s))
                    cnt[k2] += 1
                    used.update(s)
                guide[(k2, i2)] = keep
        specs.append({"lam": lam, "route_guide": child, "arc_guide": guide, "guide_free": True,
                      "blocked": list(used)})
    sols, _ = ctx.decode(specs)
    return sols


# --------------------------------------------------------------------- O4
def _unserved(inst, arcs):
    srv = {x for v in arcs.values() for s in v for x in s}
    return [int(v) for v, ok in zip(inst.info, inst.reach_info) if ok and int(v) not in srv]


def _random_move(inst, routes, arcs, rng):
    keys = list(arcs.keys())
    all_k = all_arcs(routes)
    new = {k: [list(s) for s in v] for k, v in arcs.items()}
    op = rng.choice(["insert", "new", "remove", "move", "swap", "two_opt"])
    uns = _unserved(inst, new)
    sorties = [(k, i) for k, v in new.items() for i in range(len(v))]
    if op == "insert" and uns and sorties:
        k, i = sorties[rng.integers(len(sorties))]
        v = uns[rng.integers(len(uns))]
        new[k][i].insert(int(rng.integers(len(new[k][i]) + 1)), v)
    elif op == "new" and uns:
        k = all_k[rng.integers(len(all_k))]
        v = uns[rng.integers(len(uns))]
        new.setdefault(k, []).append([v])
    elif op == "remove" and sorties:
        k, i = sorties[rng.integers(len(sorties))]
        s = new[k][i]
        s.pop(int(rng.integers(len(s))))
        if not s:
            new[k].pop(i)
    elif op in ("move", "swap") and len(sorties) >= 1:
        (k1, i1), (k2, i2) = sorties[rng.integers(len(sorties))], sorties[rng.integers(len(sorties))]
        s1, s2 = new[k1][i1], new[k2][i2]
        if op == "move":
            if s1 is s2 and len(s1) < 2:
                return None
            x = s1.pop(int(rng.integers(len(s1))))
            s2.insert(int(rng.integers(len(s2) + 1)), x)
            if not s1:
                new[k1].pop(i1)
        else:
            a, b = int(rng.integers(len(s1))), int(rng.integers(len(s2)))
            s1[a], s2[b] = s2[b], s1[a]
    elif op == "two_opt" and sorties:
        k, i = sorties[rng.integers(len(sorties))]
        s = new[k][i]
        if len(s) < 2:
            return None
        a, b = sorted(rng.choice(len(s), 2, replace=False))
        s[a:b + 1] = s[a:b + 1][::-1]
    else:
        return None
    return {k: v for k, v in new.items() if v}


def o4_local_search(ctx, sol, lam, steps=20):
    P, inst = inst_P(ctx), ctx.inst
    routes, arcs = sol.copy_plan()
    st = build_state(P, routes, arcs)
    if st is None or not st.feasible:
        return sol
    G = float(scalarize(P, st.profit, st.energy, lam, ctx.scalar))
    for _ in range(steps):
        cand = _random_move(inst, routes, arcs, ctx.rng)
        if cand is None:
            continue
        st2 = build_state(P, routes, cand)
        if st2 is None or not st2.feasible:
            continue
        G2 = float(scalarize(P, st2.profit, st2.energy, lam, ctx.scalar))
        if G2 > G + 1e-12:
            arcs, G, st = cand, G2, st2
    return Sol(P, routes, arcs, st)


def inst_P(ctx):
    return ctx.inst.P


# --------------------------------------------------------------------- đánh bóng + O5
def state_to_sol(P, st):
    arcs = {(k, idx): [list(s["seq"]) for s in lst] for k in range(P.num_trucks)
            for idx, lst in enumerate(st.sorties[k]) if lst}
    return Sol(P, st.routes, arcs, st)


def polish(st):
    """Giảm năng lượng giữ nguyên tập nút phục vụ (thứ tự bay + đổi cung). Sửa st tại chỗ."""
    lm.improve_orders(st)
    lm.reposition(st)
    st.refresh()
    return st


def polish_sols(ctx, sols):
    P = ctx.inst.P
    out = []
    for s in sols:
        if not s.feasible:
            out.append(s)
            continue
        st = build_state(P, s.routes, s.arcs)
        if st is None:
            out.append(s)
            continue
        e0 = st.energy
        polish(st)
        if st.feasible and st.energy < e0 - 1e-9:
            n = state_to_sol(P, st)
            n.lam, n.op = s.lam, s.op
            out.append(n)
        else:
            out.append(s)
    return out


def _unserved_st(P, st, reach):
    return [n for n in reach if n not in st.served]


def _best_add(st, cands):
    """Nút chưa phục vụ có tỉ lệ profit / (chi phí năng lượng biên) lớn nhất. Trả (n, ins_opt, new_opt) | None."""
    P = st.P
    best, best_r = None, -1.0
    for n in cands:
        ins = lm.best_insert(st, n)
        new = lm.best_new_sortie(st, n)
        opts = [o for o in (ins, new) if o is not None]
        if not opts:
            continue
        o = min(opts, key=lambda x: x[0])
        r = P.profit[n] / max(o[0], 1e-6)
        if r > best_r:
            best_r, best = r, (n, o is ins and ins is not None, o)
    return best


def _apply_add(st, choice):
    n, is_ins, o = choice
    if is_ins:
        lm.apply_insert(st, n, o)
    else:
        _, k, idx, s, F, E, Ef, Es = o
        st.attach(k, idx, s, F, E, Ef, Es)


def _best_drop(st):
    """Nút đang phục vụ mà xóa đi tốn ít profit nhất trên mỗi đơn vị năng lượng tiết kiệm."""
    P = st.P
    best, best_r = None, np.inf
    for k, idx, s in st.all_sorties():
        a, b, tg = st.nl[k][idx], st.nl[k][idx + 1], st.tG[k][idx]
        seq = s["seq"]
        for j, n in enumerate(seq):
            rest = seq[:j] + seq[j + 1:]
            if rest:
                _, E, _, _ = sortie_metrics(P, rest, a, b, tg)
                if E > P.theta + 1e-9:                     # lơ lửng lâu hơn -> mất cả sortie
                    lost, save = sum(P.profit[x] for x in seq), s["E"]
                else:
                    lost, save = P.profit[n], s["E"] - E
            else:
                lost, save = P.profit[n], s["E"]
            if save <= 1e-9:
                continue
            r = lost / save
            if r < best_r:
                best_r, best = r, n
    return best


def o5_pareto_ladder(ctx, parents, steps=3, n_cand=12):
    """Mỗi cha -> [cha đã đánh bóng] + tối đa `steps` con chèn dần + tối đa `steps` con xóa dần."""
    P, inst, rng = ctx.inst.P, ctx.inst, ctx.rng
    reach = [int(v) for v, ok in zip(inst.info, inst.reach_info) if ok]
    out = []
    for par in parents:
        st = build_state(P, par.routes, par.arcs)
        if st is None or not st.feasible:
            continue
        polish(st)
        base = state_to_sol(P, st)
        out.append(base)
        up = st.copy()
        for _ in range(steps):
            uns = _unserved_st(P, up, reach)
            if not uns:
                break
            if len(uns) > n_cand:
                w = np.array([P.profit[n] for n in uns], dtype=float) + 1e-6
                uns = [int(x) for x in rng.choice(uns, n_cand, replace=False, p=w / w.sum())]
            ch = _best_add(up, uns)
            if ch is None:
                break
            _apply_add(up, ch)
            up.refresh()
            lm.improve_orders(up)
            up.refresh()
            if not up.feasible:
                break
            out.append(state_to_sol(P, up))
        dn = st.copy()
        for _ in range(steps):
            n = _best_drop(dn)
            if n is None:
                break
            lm.remove_nodes(dn, [n])
            if not dn.served:
                break
            out.append(state_to_sol(P, dn))
    return out
