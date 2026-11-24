"""Môi trường MDP hai pha cho TD-TOP (Mục 2 của md/thiet_ke_mang_chinh_sach_TD-TOP.md).

Pha A: xây lộ trình xe tải (chọn nút đường bộ / END) - mask theo thời hạn T.
Pha B: với mỗi cung (theo thứ tự sự kiện) và mỗi slot UAV, chọn dãy nút thông tin rồi STOP -
       mask "dừng ngay sau v" (pin, ngân sách trễ sigma_k, ngân sách sortie b_k, mỗi nút phục vụ <= 1 lần).
Bước chỉ có đúng 1 hành động hợp lệ được áp dụng tự động và KHÔNG ghi vào quỹ đạo (không có xác suất).
Hỗ trợ teacher forcing (route_guide / arc_guide) cho toán tử O2/O3 và chưng cất elite, và tập `blocked`
(nút giữ chỗ cho sortie được giữ lại) để các bước lấy mẫu không chiếm mất.
"""

import numpy as np
import torch

from .utils import EPS, Sol

NEG = -1e9


class Env:
    def __init__(self, inst, lam, route_guide=None, arc_guide=None, guide_free=True, blocked=None, aug=0):
        self.I = inst
        self.lam = np.asarray(lam, dtype=np.float32)
        self.route_guide = route_guide
        self.arc_guide = arc_guide            # {(k, idx): [seq, ...]}
        self.guide_free = guide_free          # cung không có trong arc_guide: True = lấy mẫu, False = để trống
        self.aug = aug
        I = inst
        self.k, self.cur, self.t = 0, 0, 0.0
        self.unv = np.ones(I.num_road, dtype=bool)
        self.routes = [[] for _ in range(I.K)]
        self.phase = "A"
        self.traj = []                        # [(rec, action)]
        self.rec, self.forced = None, None
        self.repaired = False
        self.served = np.zeros(I.num_info, dtype=bool)
        self.blocked = np.zeros(I.num_info, dtype=bool)
        for v in (blocked or ()):
            self.blocked[I.info_pos[v]] = True
        self.arcs = {}
        self.arc_open = False
        self.sol = None

    # ------------------------------------------------------------------ vòng lặp quyết định
    def advance(self):
        """Áp dụng các bước bắt buộc; trả True nếu cần một quyết định (self.rec, self.forced)."""
        while True:
            if self.phase == "A":
                mask = self._mask_A()
                nv = int(mask.sum())
                if nv == 0:
                    self._repair_A()
                    self._init_B()
                    continue
                ga = self._guide_A(mask)
                if nv == 1:
                    self._apply_A(int(np.flatnonzero(mask)[0]))
                    continue
                self.rec = self._rec_A(mask)
                self.forced = ga
                return True
            if self.phase == "B":
                if not self._select_arc():
                    self.phase = "done"
                    continue
                valid = self._calc_B()
                mask_pol = valid & ~np.append(self.blocked, False)
                ga = self._guide_B(valid)
                if ga is not None:
                    if not mask_pol[ga]:
                        mask_pol = mask_pol.copy()
                        mask_pol[ga] = True
                    if mask_pol.sum() >= 2:
                        self.rec = self._rec_B(mask_pol)
                        self.forced = ga
                        return True
                    self._apply_B(ga)
                    continue
                if mask_pol.sum() == 1:        # chỉ còn STOP
                    self._apply_B(self.I.num_info)
                    continue
                self.rec = self._rec_B(mask_pol)
                self.forced = None
                return True
            return False

    @property
    def done(self):
        return self.phase == "done"

    def apply(self, action):
        """Áp dụng quyết định đang chờ (ghi vào quỹ đạo)."""
        self.traj.append((self.rec, int(action)))
        self.rec, self.forced = None, None
        (self._apply_A if self.phase == "A" else self._apply_B)(int(action))

    # ------------------------------------------------------------------ Pha A
    def _mask_A(self):
        I, k = self.I, self.k
        last = k == I.K - 1
        tnext = self.t + I.tG[self.cur, I.road]
        if last:
            M = np.where(self.unv[None, :], I.TT, -np.inf)
            np.fill_diagonal(M, -np.inf)
            lb = np.maximum(I.t_to_end, M.max(1))
        else:
            lb = I.t_to_end
        valid = self.unv & (tnext + lb <= I.T + 1e-9)
        end_ok = not (last and self.unv.any())
        return np.append(valid, end_ok)

    def _guide_A(self, mask):
        if self.route_guide is None:
            return None
        I = self.I
        for j in self.route_guide[self.k]:
            c = I.road_pos[j]
            if self.unv[c]:
                return int(c) if mask[c] else None
        return I.num_road if mask[I.num_road] else None

    def _rec_A(self, mask):
        I = self.I
        psi = np.array([self.t / I.T, self.unv.sum() / max(1, I.num_road), float(self.k == I.K - 1),
                        (I.K - self.k - 1) / I.K], dtype=np.float32)
        return {"phase": "A", "cur": self.cur, "k": self.k, "psi": psi, "mask": mask.copy()}

    def _apply_A(self, a):
        I = self.I
        if a < I.num_road:
            j = int(I.road[a])
            self.t += I.tG[self.cur, j]
            self.unv[a] = False
            self.routes[self.k].append(j)
            self.cur = j
        else:
            self.k += 1
            self.cur, self.t = 0, 0.0
            if self.k == I.K:
                self._init_B()

    def _repair_A(self):
        """Ngõ cụt (xe cuối, T quá chặt): chèn rẻ nhất các nút còn lại; đánh dấu đã sửa."""
        I = self.I
        self.repaired = True
        from .utils import route_time
        for u in [int(x) for x in I.road[self.unv]]:
            best = None
            for k in range(I.K):
                r = self.routes[k]
                nl = [0] + r + [I.end]
                base = route_time(I, r)
                for p in range(len(nl) - 1):
                    a, b = nl[p], nl[p + 1]
                    nt = base + I.tG[a, u] + I.tG[u, b] - I.tG[a, b]
                    key = (nt > I.T + 1e-9, nt)
                    if best is None or key < best[0]:
                        best = (key, k, p)
            _, k, p = best
            self.routes[k].insert(p, u)
        self.unv[:] = False
        self.k = I.K

    # ------------------------------------------------------------------ Pha B
    def _init_B(self):
        I = self.I
        self.phase = "B"
        self.nl = [[0] + r + [I.end] for r in self.routes]
        self.tg = [[I.tG[nl[i], nl[i + 1]] for i in range(len(nl) - 1)] for nl in self.nl]
        self.ptr = [0] * I.K
        self.dep = [0.0] * I.K
        self.bq = list(I.cap)
        self.sig = [I.T - sum(t) for t in self.tg]
        self.arc_open = False

    def _select_arc(self):
        I = self.I
        while not self.arc_open:
            cands = [k for k in range(I.K) if self.ptr[k] < len(self.tg[k])]
            if not cands:
                return False
            k = min(cands, key=lambda q: (self.dep[q], q))
            idx = self.ptr[k]
            self.bk, self.bidx = k, idx
            self.a, self.b = self.nl[k][idx], self.nl[k][idx + 1]
            self.tgv = self.tg[k][idx]
            self.m, self.path = 0, []
            self.D, self.Ef, self.s_sum = 0.0, 0.0, 0.0
            self.Dmax, self.delta = 0.0, 0.0
            self.arc_sorties = []
            self.arc_open = True
            if I.R[k] == 0:
                self._close_arc()
        return True

    def _close_arc(self):
        k = self.bk
        self.dep[k] += max(self.tgv, self.Dmax)
        self.ptr[k] += 1
        self.arc_open = False
        if self.arc_sorties:
            self.arcs[(k, self.bidx)] = self.arc_sorties

    def _calc_B(self):
        """Tính mask danh nghĩa (không tính `blocked`) + các đại lượng cho khóa động xi."""
        I = self.I
        c = self.path[-1] if self.path else self.a
        info, b, tg = I.info, self.b, self.tgv
        self._c = c
        Dn = self.D + I.tU[c, info] + I.s_info + I.tU[info, b]
        En = (self.Ef + I.eU[c, info] + I.eU[info, b] + I.p_srv * (self.s_sum + I.s_info)
              + I.p_hover * np.maximum(0.0, tg - Dn))
        nd = np.maximum(0.0, np.maximum(self.Dmax, Dn) - tg)
        dd = nd - self.delta
        valid = I.reach_info & ~self.served & (En <= I.theta + 1e-9) & (dd <= self.sig[self.bk] + 1e-9)
        if not self.path and self.bq[self.bk] < 1:
            valid = np.zeros_like(valid)
        self._Dn, self._En, self._dd = Dn, En, dd
        return np.append(valid, True)

    def _guided_seq(self):
        """Dãy nút mà arc_guide muốn cho slot hiện tại (None = lấy mẫu tự do)."""
        key = (self.bk, self.bidx)
        if key not in self.arc_guide:
            return None if self.guide_free else []
        lst = sorted(self.arc_guide[key], key=lambda s: s[0])
        return lst[self.m] if self.m < len(lst) else []

    def _guide_B(self, valid):
        if self.arc_guide is None:
            return None
        seq = self._guided_seq()
        if seq is None:
            return None
        I = self.I
        pos = len(self.path)
        if pos < len(seq):
            p = int(I.info_pos[seq[pos]])
            if valid[p]:
                return p
        return I.num_info          # STOP

    def _rec_B(self, mask):
        I = self.I
        c, tg = self._c, self.tgv
        if self.path:
            F = self.D + I.tU[c, self.b]
            Estop = (self.Ef + I.eU[c, self.b] + I.p_srv * self.s_sum + I.p_hover * max(0.0, tg - F))
        else:
            Estop = 0.0
        xi = np.stack([self.served.astype(np.float64), I.tU[c, I.info] / I.T, (self._Dn - self.D) / I.T,
                       (self._En - Estop) / I.theta, (I.theta - self._En) / I.theta,
                       (self.sig[self.bk] - self._dd) / I.T, I.p_info / max(1.0, I.p_info.max())], axis=1)
        xi = np.clip(xi, -3.0, 3.0).astype(np.float32)
        k = self.bk
        psi = np.array([self.D / I.T, (self.Ef + I.p_srv * self.s_sum) / I.theta, tg / I.T, self.delta / I.T,
                        self.sig[k] / I.T, self.bq[k] / max(1, I.cap[k]), self.served.sum() / max(1, I.num_info),
                        self.bidx / max(1, len(self.tg[k]))], dtype=np.float32)
        return {"phase": "B", "i": self.a, "j": self.b, "last": c, "m": min(self.m, 15), "k": k,
                "psi": psi, "xi": xi, "mask": mask.copy()}

    def _apply_B(self, a):
        I = self.I
        if a < I.num_info:
            v = int(I.info[a])
            c = self._c if self.path else self.a
            c = self.path[-1] if self.path else self.a
            self.D += I.tU[c, v] + I.service[v]
            self.Ef += I.eU[c, v]
            self.s_sum += I.service[v]
            self.path.append(v)
            self.served[a] = True
            self.blocked[a] = False
            return
        # STOP
        if not self.path:
            self._close_arc()
            return
        c, b, tg = self.path[-1], self.b, self.tgv
        F = self.D + I.tU[c, b]
        self.arc_sorties.append(list(self.path))
        self.Dmax = max(self.Dmax, F)
        nd = max(0.0, self.Dmax - tg)
        self.sig[self.bk] -= nd - self.delta
        self.delta = nd
        self.bq[self.bk] -= 1
        self.m += 1
        self.path = []
        self.D = self.Ef = self.s_sum = 0.0
        if self.m >= I.R[self.bk] or self.bq[self.bk] <= 0:
            self._close_arc()

    # ------------------------------------------------------------------ kết quả
    def solution(self):
        if self.sol is None:
            self.sol = Sol(self.I.P, self.routes, self.arcs)
        return self.sol


# ====================================================================
# Gom bước thành tensor
# ====================================================================
def collate(recs, aug, lam, phase):
    t = lambda x, dt: torch.as_tensor(np.asarray(x), dtype=dt)
    bt = {"phase": phase, "aug": t(aug, torch.long), "lam": t(np.asarray(lam), torch.float32),
          "k": t([r["k"] for r in recs], torch.long), "psi": t(np.stack([r["psi"] for r in recs]), torch.float32),
          "mask": t(np.stack([r["mask"] for r in recs]), torch.bool)}
    if phase == "A":
        bt["cur"] = t([r["cur"] for r in recs], torch.long)
    else:
        for key in ("i", "j", "last", "m"):
            bt[key] = t([r[key] for r in recs], torch.long)
        bt["xi"] = t(np.stack([r["xi"] for r in recs]), torch.float32)
    return bt


def pending_batch(envs, phase):
    es = [e for e in envs if e.rec is not None and e.phase == phase]
    if not es:
        return es, None
    return es, collate([e.rec for e in es], [e.aug for e in es], [e.lam for e in es], phase)


def stack_traj(envs, phase, env_ids=None):
    """Mọi bước đã ghi của `phase` trong các env -> (tensor batch, hành động, chỉ số env)."""
    recs, acts, eids, augs, lams = [], [], [], [], []
    for n, e in enumerate(envs):
        for rec, a in e.traj:
            if rec["phase"] == phase:
                recs.append(rec)
                acts.append(a)
                eids.append(n if env_ids is None else env_ids[n])
                augs.append(e.aug)
                lams.append(e.lam)
    if not recs:
        return None
    bt = collate(recs, augs, lams, phase)
    return bt, torch.as_tensor(acts, dtype=torch.long), torch.as_tensor(eids, dtype=torch.long)


@torch.no_grad()
def run_batch(policy, enc, envs, greedy=False, chunk=512):
    """Chạy đồng thời các env cho tới khi xong; mỗi vòng gom các quyết định theo pha để gọi mạng một lần."""
    while True:
        pend = [e for e in envs if not e.done and e.advance()]
        if not pend:
            break
        free = []
        for e in pend:
            if e.forced is not None:
                e.apply(e.forced)
            else:
                free.append(e)
        for phase in ("A", "B"):
            es = [e for e in free if e.phase == phase]
            for s in range(0, len(es), chunk):
                part = es[s:s + chunk]
                bt = collate([e.rec for e in part], [e.aug for e in part], [e.lam for e in part], phase)
                acts = policy.act(enc, bt, greedy=greedy)
                for e, a in zip(part, acts.tolist()):
                    e.apply(a)
    return envs
