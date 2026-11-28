"""Môi trường MDP cho PPO: xây một lời giải TD-TOP hoàn chỉnh, từng quyết định một.

Không gian hành động chung: chỉ số nút 0..n-1 + 2 hành động đặc biệt
    SPEC0 = n     giai đoạn 1: kết thúc xe hiện tại (sang xe sau)   | giai đoạn 2: đóng sortie đang xây (STOP)
    SPEC1 = n+1   giai đoạn 1: (không dùng)                          | giai đoạn 2: sang cung kế tiếp (NEXT_ARC)

Giai đoạn 1 - lộ trình xe tải: xe k đứng ở `cur`, chọn nút đường chưa thăm (mask: còn kịp về kho trước T)
    hoặc SPEC0. Xe cuối phải thăm hết nút còn lại; ngõ cụt -> chèn rẻ nhất và đánh dấu repaired.
Giai đoạn 2 - sortie UAV: duyệt lần lượt các cung (k, idx) của các lộ trình; trên mỗi cung xây tối đa R_k sortie
    (tổng số sortie của xe <= |Q_k|). Mỗi bước: thêm một nút thông tin vào sortie đang mở
    (mask: chưa phục vụ, pin E <= theta, xe vẫn về kịp T sau khi chờ UAV), hoặc STOP (đóng sortie),
    hoặc NEXT_ARC (khi sortie đang mở rỗng).

Phần thưởng theo sở thích lambda = (w1, w2) - dày, tổng có trọng số:
    r = w1 * dProfit / F1_REF - w2 * dEnergy / F2_REF   (lời giải không khả thi: thêm -1 ở cuối)
    Tín hiệu theo từng bước -> học nhanh, ổn định. Về lý thuyết chỉ nhắm điểm Pareto trên bao lồi,
    nhưng policy gần đúng + lấy mẫu vẫn sinh nhiều điểm không lồi (archive giữ lại).
Bước chỉ có đúng một hành động hợp lệ được áp dụng tự động, không ghi vào quỹ đạo; phần thưởng của chúng
được cộng vào bước quyết định gần nhất trước đó.
"""

import numpy as np

from .common import EPS, State, sortie_metrics

DYN, GLOB = 6, 12
INFEASIBLE_PENALTY = 1.0


class Env:
    """route_guide / arc_guide / guide_free / blocked: teacher forcing cho toán tử O2/O3 của NSGA-II
    (cùng ngữ nghĩa với rl/core/environment.Env). Bước được hướng dẫn áp dụng thẳng, không hỏi mạng, không ghi
    vào quỹ đạo -> chỉ dùng khi giải (record=False), không dùng khi huấn luyện.
        route_guide: [[nút đường của xe k], ...] - đi theo nếu mask cho phép, ngược lại để mạng chọn
        arc_guide  : {(k, idx): [seq, ...]} - sortie giữ lại trên cung idx của xe k
        guide_free : cung không có trong arc_guide: True = mạng chọn tự do, False = bỏ trống
        blocked    : nút thông tin giữ chỗ cho sortie được giữ (mạng không được chọn)
    imitate=True: bước được hướng dẫn mà có >= 2 hành động hợp lệ KHÔNG áp dụng thẳng, mà dừng lại với self.obs
        và self.forced = hành động hướng dẫn -> người gọi step(self.forced) để ghi quỹ đạo của một lời giải cho
        trước (học bắt chước elite, xem nsga_ctx.replay)."""

    def __init__(self, inst, lam, aug=0, route_guide=None, arc_guide=None, guide_free=True, blocked=None,
                 imitate=False):
        self.imitate = imitate
        self.forced = None
        self.I, self.P = inst, inst.P
        P = self.P
        self.n = inst.n
        self.lam = np.asarray(lam, dtype=float)
        self.aug = int(aug)
        self.route_guide = route_guide
        self.arc_guide = arc_guide
        self.guide_free = guide_free
        self.blocked = np.zeros(self.n, bool)
        for v in (blocked or ()):
            self.blocked[int(v)] = True
        self.traj = []
        self.obs = None
        self.done = False
        self.repaired = False
        self._acc = 0.0
        # giai đoạn 1
        self.stage = 1
        self.k, self.cur, self.t = 0, 0, 0.0
        self.routes = [[] for _ in range(P.num_trucks)]
        self.unvisited = np.zeros(self.n, bool)
        self.unvisited[inst.road] = True
        # giai đoạn 2
        self.st = None
        self._advance()

    # ------------------------------------------------------------------ vòng quyết định
    def step(self, action, logp=0.0, value=0.0, record=True):
        """Áp dụng hành động của agent cho quyết định đang chờ (self.obs)."""
        if record:
            self.traj.append({**self.obs, "act": int(action), "logp": float(logp), "value": float(value), "r": 0.0})
        self._acc = 0.0
        self._apply(int(action))
        self._advance()
        if record:
            self.traj[-1]["r"] += self._acc

    def _advance(self):
        """Áp dụng các bước bắt buộc cho tới khi cần một quyết định thật (self.obs) hoặc kết thúc."""
        self.obs, self.forced = None, None
        while not self.done:
            mask = self._mask()
            g = self._guide(mask)
            if g is not None:                      # teacher forcing
                if self.imitate and int(mask.sum()) > 1:
                    self.obs, self.forced = self._observe(mask), g
                    return
                self._apply(g)
                continue
            if self.stage == 2 and self.blocked.any():
                mask[:self.n] &= ~self.blocked
            n_legal = int(mask.sum())
            if n_legal == 0:                       # chỉ xảy ra ở giai đoạn 1 (ngõ cụt của xe cuối)
                self._repair()
                continue
            if n_legal == 1:
                self._apply(int(np.flatnonzero(mask)[0]))
                continue
            self.obs = self._observe(mask)
            return

    # ------------------------------------------------------------------ teacher forcing
    def _guide(self, mask):
        """Hành động hướng dẫn cho quyết định hiện tại (None = để mạng chọn)."""
        n = self.n
        if self.stage == 1:
            if self.route_guide is None:
                return None
            for j in self.route_guide[self.k]:
                if self.unvisited[j]:
                    return int(j) if mask[j] else None
            return n if mask[n] else None
        if self.arc_guide is None:
            return None
        k, idx = self.arcs[self.ai]
        if (k, idx) not in self.arc_guide:
            if self.guide_free:
                return None
            seq = []
        else:
            lst = sorted(self.arc_guide[(k, idx)], key=lambda s: s[0])
            seq = lst[self.m] if self.m < len(lst) else []
        pos = len(self.seq)
        if pos < len(seq) and mask[seq[pos]]:
            return int(seq[pos])
        return n if self.seq else n + 1            # đóng sortie / sang cung kế tiếp

    # ------------------------------------------------------------------ mask
    def _mask(self):
        return self._mask1() if self.stage == 1 else self._mask2()

    def _mask1(self):
        I, P, n = self.I, self.P, self.n
        m = np.zeros(n + 2, bool)
        m[:n] = self.unvisited & (self.t + I.tt[self.cur] + I.tt[:, P.end] <= P.T + EPS)
        last = self.k == P.num_trucks - 1
        m[n] = (not self.unvisited.any()) or not last
        return m

    def _arc(self):
        k, idx = self.arcs[self.ai]
        nl = self.st.nl[k]
        return k, idx, nl[idx], nl[idx + 1], self.st.tG[k][idx]

    def _cand2(self):
        """F, E, thời gian dùng của xe nếu thêm từng nút vào sortie đang mở."""
        I, P = self.I, self.P
        k, idx, a, b, tg = self._arc()
        l = self.seq[-1] if self.seq else a
        F = self.Fp + I.ut[l] + I.sv + I.ut[:, b]
        E = (self.Efp + I.ee[l] + I.ee[:, b] + P.p_srv * (self.ssum + I.sv)
             + P.p_hover * np.maximum(0.0, tg - F))
        dur = self.st.dur[k][idx]
        used = self.st.used[k] - dur + np.maximum(dur, F)
        return F, E, used

    def _mask2(self):
        P, n = self.P, self.n
        m = np.zeros(n + 2, bool)
        _, E, used = self._cand2()
        m[:n] = self.avail & (E <= P.theta + 1e-6) & (used <= P.T + EPS)
        m[n] = len(self.seq) > 0
        m[n + 1] = len(self.seq) == 0
        return m

    # ------------------------------------------------------------------ chuyển trạng thái
    def _apply(self, a):
        if self.stage == 1:
            self._apply1(a)
        else:
            self._apply2(a)

    def _apply1(self, a):
        I, P, n = self.I, self.P, self.n
        if a < n:
            self.routes[self.k].append(a)
            self.t += I.tt[self.cur, a]
            self.cur = a
            self.unvisited[a] = False
        elif self.unvisited.any() and self.k < P.num_trucks - 1:
            self.k, self.cur, self.t = self.k + 1, 0, 0.0
        else:
            self._start_stage2()

    def _repair(self):
        """Ngõ cụt của xe cuối: chèn rẻ nhất các nút còn lại (có thể vi phạm T)."""
        I, P = self.I, self.P
        r = self.routes[self.k]
        for v in np.flatnonzero(self.unvisited):
            nl = [0] + r + [P.end]
            cost = [I.tt[nl[i], v] + I.tt[v, nl[i + 1]] - I.tt[nl[i], nl[i + 1]] for i in range(len(nl) - 1)]
            r.insert(int(np.argmin(cost)), int(v))
        self.unvisited[:] = False
        self.repaired = True
        self._start_stage2()

    def _start_stage2(self):
        P = self.P
        self.st = State(P, self.routes)
        self.stage = 2
        if not self.st.feasible:
            self._acc -= INFEASIBLE_PENALTY
            self.done = True
            return
        self.arcs = [(k, idx) for k in range(P.num_trucks) if P.R[k] > 0 and P.cap[k] > 0
                     for idx in range(len(self.st.tG[k]))]
        self.avail = self.I.reach.copy()
        self.ai, self.m = 0, 0
        self._reset_seq()
        if not self.arcs or not self.avail.any():
            self.done = True

    def _reset_seq(self):
        self.seq, self.Fp, self.Efp, self.ssum = [], 0.0, 0.0, 0.0

    def _apply2(self, a):
        I, P, n = self.I, self.P, self.n
        k, idx, la, lb, tg = self._arc()
        w1, w2 = self.lam
        if a < n:
            l = self.seq[-1] if self.seq else la
            self.Fp += I.ut[l, a] + I.sv[a]
            self.Efp += I.ee[l, a]
            self.ssum += I.sv[a]
            self.seq.append(a)
            self.avail[a] = False
            self.blocked[a] = False
            self._acc += w1 * I.profit[a] / P.f1_ref
        elif a == n:                                   # STOP: đóng sortie
            F, E, Ef, Es = sortie_metrics(P, self.seq, la, lb, tg)
            self.st.attach(k, idx, self.seq, F, E, Ef, Es)
            self._acc -= w2 * E / P.f2_ref
            self._reset_seq()
            self.m += 1
            if self.m >= P.R[k] or self.st.cnt[k] >= P.cap[k]:
                self._next_arc()
        else:                                          # NEXT_ARC
            self._next_arc()

    def _next_arc(self):
        P = self.P
        self.ai += 1
        self.m = 0
        while self.ai < len(self.arcs) and self.st.cnt[self.arcs[self.ai][0]] >= P.cap[self.arcs[self.ai][0]]:
            self.ai += 1
        if self.ai >= len(self.arcs) or not self.avail.any():
            self.done = True

    # ------------------------------------------------------------------ quan sát
    def _observe(self, mask):
        I, P, n = self.I, self.P, self.n
        T = max(P.T, 1e-9)
        dyn = np.zeros((n, DYN), np.float32)
        glob = np.zeros(GLOB, np.float32)
        glob[10:12] = self.lam
        if self.stage == 1:
            cur, a, b = self.cur, self.cur, P.end
            dyn[:, 0] = self.unvisited
            dyn[:, 1] = I.tt[self.cur] / T
            dyn[:, 2] = (self.t + I.tt[self.cur] + I.tt[:, P.end]) / T
            dyn[:, 5] = ~self.unvisited
            glob[1] = self.k / max(1, P.num_trucks)
            glob[2] = self.t / T
            glob[9] = self.unvisited.sum() / max(1, P.num_road)
        else:
            k, idx, a, b, tg = self._arc()
            cur = self.seq[-1] if self.seq else a
            _, E, used = self._cand2()
            slack = max(T - self.st.used[k], 1e-9)
            dyn[:, 0] = self.avail
            dyn[:, 1] = I.ut[cur] / T * 10.0
            dyn[:, 2] = np.clip(E / P.theta, 0.0, 2.0)
            dyn[:, 3] = np.clip((used - self.st.used[k]) / slack, 0.0, 2.0)
            dyn[:, 5] = ~self.avail & I.is_info
            glob[0] = 1.0
            glob[1] = k / max(1, P.num_trucks)
            glob[2] = self.st.used[k] / T
            glob[3] = self.ai / max(1, len(self.arcs))
            glob[4] = self.m / max(1, P.R[k])
            glob[5] = min(1.0, self.st.cnt[k] / max(1, P.cap[k]))
            glob[6] = len(self.seq) / 5.0
            if self.seq:
                glob[7] = sortie_metrics(P, self.seq, a, b, tg)[1] / P.theta
            glob[8] = float((I.profit * self.avail).sum() / P.f1_ref)
        dyn[:, 4] = mask[:n]
        return {"stage": self.stage - 1, "cur": int(cur), "a": int(a), "b": int(b), "aug": self.aug,
                "dyn": dyn, "glob": glob, "mask": mask}

    # ------------------------------------------------------------------ kết quả
    @property
    def feasible(self):
        return self.st is not None and self.st.feasible

    def objectives(self):
        """(f1, f2) gốc; None nếu không khả thi."""
        return (self.st.profit, self.st.energy) if self.feasible else None
