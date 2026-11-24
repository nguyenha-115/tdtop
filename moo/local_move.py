"""Bước di chuyển cục bộ.

Phần 1 - trên State (dùng trong bộ giải mã): đặt sortie mới, chèn nút vào sortie có sẵn, xóa nút,
         tối ưu thứ tự bay (2-opt / or-opt), chuyển sortie sang cung rẻ hơn.
         Mọi hàm giữ ràng buộc pin, R_k sortie/cung, |Q_k| sortie/xe và không làm xe vượt T khi chèn.
Phần 2 - bộ giải mã `decode(P, x)`: chromosome random-key -> State (tất định, không dùng số ngẫu nhiên).
Phần 3 - lân cận trên chromosome (bật / tắt / đổi nút thông tin, đổi hành trình xe tải) và tìm kiếm cục bộ
         Lamarck dùng chung cho mọi thuật toán trong moo_algorithm.
"""

import numpy as np

from config import DECODER, EPS, INFEASIBLE_OFFSET, KEY_HIGH, KEY_LOW, KEY_MIN_ACTIVE, MAX_STEP, W_T
from problem import Problem, State, sortie_metrics


# ====================================================================
# Phần 1 - thao tác trên State
# ====================================================================
def arc_options(st: State, seq):
    """Mọi cách đặt một sortie MỚI bay qua `seq` (cả hai chiều) lên một cung còn UAV rảnh.
    Trả về list (cost, k, idx, seq, F, E, Ef, Es) với cost = E + W_T * (thời gian xe bị kéo dài)."""
    P = st.P
    out = []
    orients = [seq] if len(seq) == 1 else [seq, seq[::-1]]
    for s in orients:
        for k in range(P.num_trucks):
            if st.cnt[k] >= P.cap[k]:                       # hết |Q_k|
                continue
            nl = st.nl[k]
            for idx in range(len(nl) - 1):
                if len(st.sorties[k][idx]) >= P.R[k]:       # hết UAV trên cung (50)-(51)
                    continue
                F, E, Ef, Es = sortie_metrics(P, s, nl[idx], nl[idx + 1], st.tG[k][idx])
                if E > P.theta + EPS:                       # pin (60)
                    continue
                delta = max(0.0, F - st.dur[k][idx])
                if delta > EPS and st.used[k] + delta > P.T + EPS:   # khung thời gian (58)
                    continue
                out.append((E + W_T * delta, k, idx, list(s), F, E, Ef, Es))
    return out


def best_new_sortie(st: State, n):
    """Sortie mới rẻ nhất chỉ phục vụ nút n (None nếu không đặt được)."""
    opts = arc_options(st, [n])
    return min(opts, key=lambda o: o[0]) if opts else None


def best_insert(st: State, n):
    """Vị trí rẻ nhất để chèn n vào một sortie có sẵn.
    Trả về (cost, k, idx, si, pos, nF, nE, nEf, nEs) hoặc None."""
    P = st.P
    ut, ee = P.uav_time, P.uav_energy
    sn = P.service[n]
    pn = P.p_srv * sn
    th, ph, Tl = P.theta + EPS, P.p_hover, P.T + EPS
    best = None
    for k in range(P.num_trucks):
        nl = st.nl[k]
        used = st.used[k]
        for idx, lst in enumerate(st.sorties[k]):
            if not lst:
                continue
            a, b, tg, d0 = nl[idx], nl[idx + 1], st.tG[k][idx], st.dur[k][idx]
            for si, s in enumerate(lst):
                F, E, Ef, Es = s["F"], s["E"], s["Ef"], s["Es"]
                # cận dưới: năng lượng bay + phục vụ hiện tại cộng phục vụ n đã vượt pin
                if Ef + Es + pn > th:
                    continue
                seq = s["seq"]
                oth = tg                                       # thời gian cung nếu không tính sortie này
                for j, o in enumerate(lst):
                    if j != si and o["F"] > oth:
                        oth = o["F"]
                prev = a
                L = len(seq)
                for pos in range(L + 1):
                    nxt = seq[pos] if pos < L else b
                    nF = F + ut[prev][n] + ut[n][nxt] - ut[prev][nxt] + sn
                    nEf = Ef + ee[prev][n] + ee[n][nxt] - ee[prev][nxt]
                    nEs = Es + pn
                    hov = tg - nF
                    nE = nEf + nEs + (ph * hov if hov > 0 else 0.0)
                    if nE <= th:
                        delta = (nF if nF > oth else oth) - d0
                        if delta <= EPS or used + delta <= Tl:
                            cost = nE - E + (W_T * delta if delta > 0 else 0.0)
                            if best is None or cost < best[0]:
                                best = (cost, k, idx, si, pos, nF, nE, nEf, nEs)
                    if pos < L:
                        prev = seq[pos]
    return best


def apply_insert(st: State, n, opt):
    _, k, idx, si, pos, nF, nE, nEf, nEs = opt
    s = st.sorties[k][idx][si]
    s["seq"].insert(pos, n)
    st.energy += nE - s["E"]
    s["F"], s["E"], s["Ef"], s["Es"] = nF, nE, nEf, nEs
    d = max([st.tG[k][idx]] + [o["F"] for o in st.sorties[k][idx]])
    st.used[k] += d - st.dur[k][idx]
    st.dur[k][idx] = d
    st.served.add(n)
    st.profit += st.P.profit[n]


def insert_node(st: State, n) -> bool:
    """Phục vụ nút n theo cách rẻ nhất (chèn vào sortie có sẵn hoặc mở sortie mới). False nếu không thể."""
    if n in st.served:
        return False
    ins = best_insert(st, n)
    new = best_new_sortie(st, n)
    if ins is None and new is None:
        return False
    if new is None or (ins is not None and ins[0] <= new[0]):
        apply_insert(st, n, ins)
        return True
    _, k, idx, s, F, E, Ef, Es = new
    st.attach(k, idx, s, F, E, Ef, Es)
    return True


def remove_nodes(st: State, nodes):
    """Bỏ phục vụ các nút `nodes`; sortie rỗng hoặc (do lơ lửng lâu hơn) vượt pin thì bị xóa."""
    P = st.P
    nodes = set(nodes)
    for k in range(P.num_trucks):
        for idx, lst in enumerate(st.sorties[k]):
            if not lst:
                continue
            a, b, tg = st.nl[k][idx], st.nl[k][idx + 1], st.tG[k][idx]
            newlst = []
            for s in lst:
                if any(n in nodes for n in s["seq"]):
                    seq = [n for n in s["seq"] if n not in nodes]
                    if not seq:
                        continue
                    F, E, Ef, Es = sortie_metrics(P, seq, a, b, tg)
                    if not E <= P.theta + EPS:
                        continue
                    s.update(seq=seq, F=F, E=E, Ef=Ef, Es=Es)
                    newlst.append(s)
                else:
                    newlst.append(s)
            st.sorties[k][idx] = newlst
    st.refresh()


def opt_order(P: Problem, seq, a, b, tg):
    """Thứ tự bay tốt nhất (2-opt + or-opt đoạn 1..3, có đảo chiều) cho sortie a -> seq -> b."""
    def J(s):
        F, E, _, _ = sortie_metrics(P, s, a, b, tg)
        return E + W_T * max(0.0, F - tg)

    seq = list(seq)
    n = len(seq)
    if n < 2:
        return seq
    cur = J(seq)
    improved, guard = True, 0
    while improved and guard < 8:
        improved = False
        guard += 1
        # 2-opt: đảo đoạn seq[i..j]
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                c = J(cand)
                if c < cur - 1e-9:
                    seq, cur, improved = cand, c, True
        # or-opt: dời đoạn dài L (giữ hoặc đảo chiều) sang vị trí khác
        for L in (1, 2, 3):
            if n <= L:
                continue
            for i in range(n - L + 1):
                seg, rest = seq[i:i + L], seq[:i] + seq[i + L:]
                for j in range(len(rest) + 1):
                    if j == i:
                        continue
                    for s2 in ((seg, seg[::-1]) if L > 1 else (seg,)):
                        cand = rest[:j] + s2 + rest[j:]
                        c = J(cand)
                        if c < cur - 1e-9:
                            seq, cur, improved = cand, c, True
    return seq


def improve_orders(st: State):
    """Áp opt_order cho mọi sortie có >= 2 nút nếu giảm năng lượng mà không làm xe vượt T."""
    P = st.P
    for k, idx, s in st.all_sorties():
        if len(s["seq"]) < 2:
            continue
        a, b, tg = st.nl[k][idx], st.nl[k][idx + 1], st.tG[k][idx]
        new = opt_order(P, s["seq"], a, b, tg)
        if new == s["seq"]:
            continue
        F, E, Ef, Es = sortie_metrics(P, new, a, b, tg)
        if E >= s["E"] - 1e-9 or E > P.theta + EPS:
            continue
        d_new = max([tg, F] + [o["F"] for o in st.sorties[k][idx] if o is not s])
        delta = d_new - st.dur[k][idx]
        if delta > EPS and st.used[k] + delta > P.T + EPS:
            continue
        s.update(seq=new, F=F, E=E, Ef=Ef, Es=Es)
        st.refresh()


def reposition(st: State):
    """Chuyển từng sortie sang cung (của bất kỳ xe nào) cho chi phí E + W_T * kéo dài thấp hơn."""
    P = st.P
    for k, idx, s in st.all_sorties():
        a, b, tg = st.nl[k][idx], st.nl[k][idx + 1], st.tG[k][idx]
        F0, E0, _, _ = sortie_metrics(P, s["seq"], a, b, tg)
        st.detach(k, idx, s)
        cur_cost = E0 + W_T * max(0.0, F0 - st.dur[k][idx])
        opts = arc_options(st, s["seq"])
        best = min(opts, key=lambda o: o[0]) if opts else None
        if best is not None and best[0] < cur_cost - 1e-7:
            _, k2, idx2, s2, F, E, Ef, Es = best
            st.sorties[k2][idx2].append({"seq": s2, "F": F, "E": E, "Ef": Ef, "Es": Es})
        else:
            st.sorties[k][idx].append(s)
        st.refresh()


def improve(st: State):
    if DECODER["opt_order"]:
        improve_orders(st)
    if DECODER["reposition"]:
        reposition(st)


def route_2opt(P: Problem, route, max_pass=50):
    """2-opt (khoảng cách Manhattan) cho lộ trình xe 0 -> route -> 0'."""
    n = len(route)
    if n < 2:
        return list(route)
    D = P.truck_dist_np
    ext = np.array([0] + list(route) + [P.end])
    for _ in range(max_pass):
        improved = False
        for i in range(1, n):
            a, b = ext[i - 1], ext[i]
            c, d = ext[i + 1:n + 1], ext[i + 2:n + 2]
            gain = D[a, b] + D[c, d] - D[a, c] - D[b, d]
            j = int(np.argmax(gain))
            if gain[j] > 1e-9:
                j += i + 1
                ext[i:j + 1] = ext[i:j + 1][::-1].copy()
                improved = True
        if not improved:
            break
    return [int(v) for v in ext[1:-1]]


# ====================================================================
# Phần 2 - bộ giải mã / mã hóa
# ====================================================================
def decode(P: Problem, x) -> State:
    """chromosome -> State:
    1. khóa nút đường -> phân xe + thứ tự thăm (rồi 2-opt);
    2. chèn tham lam các nút thông tin được chọn theo thứ tự khóa giảm dần;
    3. cải thiện (thứ tự bay, đổi cung) rồi thử chèn lại các nút chưa chèn được."""
    x = np.asarray(x, dtype=float)
    routes = P.keys_to_routes(P.road_keys(x))
    if DECODER["route_2opt"]:
        routes = [route_2opt(P, r) for r in routes]
    st = State(P, routes)
    chosen = P.selected_info(P.info_keys(x))
    failed = [n for n in chosen if not insert_node(st, n)]
    improve(st)
    if DECODER["second_pass"] and failed:
        added = False
        for n in failed:
            added |= insert_node(st, n)
        if added:
            improve(st)
    st.refresh()
    return st


def encode(P: Problem, st: State, x_ref=None):
    """State -> chromosome (dùng cho Lamarck): khóa đường theo lộ trình, khóa thông tin dương <=> nút được phục vụ.
    Độ lớn khóa thông tin lấy từ x_ref (nếu có) để giữ thứ tự ưu tiên."""
    x = np.empty(P.n_var)
    x[:P.num_road] = P.routes_to_keys(st.routes)
    ik = np.array(P.info_keys(x_ref), dtype=float) if x_ref is not None else np.zeros(P.num_info)
    for j, n in enumerate(P.info_ids):
        v = abs(ik[j]) if abs(ik[j]) >= KEY_MIN_ACTIVE else KEY_MIN_ACTIVE
        x[P.num_road + j] = v if n in st.served else -v
    return x


# ====================================================================
# Phần 3 - lân cận trên chromosome + tìm kiếm cục bộ
# ====================================================================
def _pick(rng, items, weights, m):
    """Chọn m phần tử không lặp, xác suất tỉ lệ với weights."""
    items = list(items)
    if not items:
        return []
    m = min(m, len(items))
    w = np.asarray(weights, dtype=float) + 1e-12
    idx = rng.choice(len(items), size=m, replace=False, p=w / w.sum())
    return [items[i] for i in np.atleast_1d(idx)]


def move_add(P: Problem, x, rng, m=1):
    """Bật m nút thông tin đang tắt (ưu tiên nút có giá trị/năng lượng cao)."""
    off = [j for j, n in enumerate(P.info_ids) if P.reachable[n] and x[P.num_road + j] <= 0]
    for j in _pick(rng, off, [P.score[P.info_ids[j]] for j in off], m):
        x[P.num_road + j] = rng.uniform(KEY_MIN_ACTIVE, KEY_HIGH)
    return x


def move_drop(P: Problem, x, rng, m=1):
    """Tắt m nút thông tin đang bật (ưu tiên nút có giá trị/năng lượng thấp)."""
    on = [j for j, n in enumerate(P.info_ids) if P.reachable[n] and x[P.num_road + j] > 0]
    for j in _pick(rng, on, [1.0 / (P.score[P.info_ids[j]] + 1e-6) for j in on], m):
        x[P.num_road + j] = -rng.uniform(KEY_MIN_ACTIVE, KEY_HIGH)
    return x


def move_swap(P: Problem, x, rng):
    """Đổi khóa của hai nút thông tin (thường là một nút bật với một nút tắt)."""
    if P.num_info < 2:
        return x
    on = [j for j in range(P.num_info) if x[P.num_road + j] > 0]
    off = [j for j in range(P.num_info) if x[P.num_road + j] <= 0]
    if on and off and rng.random() < 0.7:
        i, j = on[rng.randint(len(on))], off[rng.randint(len(off))]
    else:
        i, j = rng.choice(P.num_info, 2, replace=False)
    a, b = P.num_road + i, P.num_road + j
    x[a], x[b] = x[b], x[a]
    return x


def move_route(P: Problem, x, rng):
    """Đổi hành trình xe tải: hoán đổi hai nút / chuyển một nút sang xe khác / đảo một đoạn (2-opt)."""
    nr = P.num_road
    if nr == 0:
        return x
    op = rng.randint(3)
    if op == 0 and nr >= 2:
        i, j = rng.choice(nr, 2, replace=False)
        x[i], x[j] = x[j], x[i]
    elif op == 1:
        i = rng.randint(nr)
        k = rng.randint(P.num_trucks)
        lo = KEY_LOW + k * P.truck_width
        x[i] = rng.uniform(lo + 1e-6, lo + P.truck_width - 1e-6)
    else:
        routes = P.keys_to_routes(x[:nr])
        cand = [r for r in routes if len(r) >= 2]
        if cand:
            r = cand[rng.randint(len(cand))]
            i, j = sorted(rng.choice(len(r), 2, replace=False))
            seg = [n - 1 for n in r[i:j + 1]]
            vals = x[seg].copy()
            x[seg] = vals[::-1]
    return x


def random_move(P: Problem, x, rng, max_step=MAX_STEP):
    """Một bước lân cận ngẫu nhiên (trả về bản sao)."""
    x = np.array(x, dtype=float)
    r = rng.random()
    m = int(rng.randint(1, max_step + 1))
    if r < 0.3:
        move_add(P, x, rng, m)
    elif r < 0.6:
        move_drop(P, x, rng, m)
    elif r < 0.85:
        move_swap(P, x, rng)
    else:
        move_route(P, x, rng)
    return x


def objectives_of(P: Problem, st: State):
    """Vector mục tiêu chuẩn hóa (cực tiểu hóa) giống utils.fitness_of_state."""
    v = st.violation
    if v > EPS:
        pen = 1.0 + INFEASIBLE_OFFSET + v
        return (pen, pen)
    return P.normalize(st.profit, st.energy)


def dominates(a, b):
    return a[0] <= b[0] and a[1] <= b[1] and (a[0] < b[0] or a[1] < b[1])


def local_search(P: Problem, x, rng, steps=10):
    """Leo đồi Pareto: nhận bước lân cận nếu trội lời giải hiện tại. Trả về (x, objectives)."""
    x = np.array(x, dtype=float)
    cur = objectives_of(P, decode(P, x))
    for _ in range(steps):
        y = random_move(P, x, rng)
        o = objectives_of(P, decode(P, y))
        if dominates(o, cur):
            x, cur = y, o
    return x, cur
