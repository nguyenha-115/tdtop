"""Cắm chính sách PPO vào KL-NSGA-II-RL (rl/core/nsga2_kl_rl.py).

PPOCtx thay rl/core/operators.Ctx: cùng giao diện decode(specs, greedy) -> (list Sol, list Env), nên các toán tử
O1 (lấy mẫu), O2 (phá-xây), O3 (lai ghép lộ trình), O4/O5 (tìm kiếm cục bộ) dùng nguyên như với mạng POMO.
Teacher forcing của O2/O3 do rl/ppo/env.Env đảm nhận (route_guide, arc_guide, guide_free, blocked).
Lời giải được dựng lại bằng rl/core/utils.Sol (simulator moo là trọng tài cuối cùng) như với POMO.

Chế độ online (thích nghi theo instance, kiểu EAS - Efficient Active Search):
    encoder + thân mạng ĐÓNG BĂNG, chỉ tinh chỉnh adapter có điều kiện lambda (PPONet.adapter, khởi tạo = hàm
    đồng nhất). Mỗi U thế hệ:
      1. học bắt chước elite: phát lại (teacher forcing, Env(imitate=True)) các lời giải trong archive với lambda
         được gán cho chúng -> NLL các hành động của elite. Elite đã qua đánh bóng / O5 nên mạng học được cả phần
         cải thiện mà tìm kiếm cục bộ tìm ra (chưng cất tìm kiếm vào chính sách).
      2. PPO-clip trên mẫu mới: n_lam lambda x n_traj quỹ đạo, lợi thế = R - trung bình R cùng lambda (baseline
         chung kiểu POMO, không cần critic), chuẩn hóa theo độ lệch chuẩn.
      L = L_PPO + alpha * L_NLL;  các mẫu mới cũng được trả về làm con cho NSGA-II (không lãng phí lượt giải mã).
"""

import numpy as np
import torch

from rl.core.operators import Ctx
from rl.core.utils import Sol, assign_lambda

from .common import Instance
from .env import Env
from .model import collate, encode_instance, rollout


class PPOCtx(Ctx):
    def __init__(self, inst, net, rng, Lam, n_aug=8, scalar="weighted", chunk=1024):
        self.pinst = Instance(inst.P.name)
        H = encode_instance(net, self.pinst, n_aug)
        super().__init__(inst, net, H, rng, Lam, n_aug, scalar)
        self.chunk = chunk
        self.opt = None

    def to_sol(self, e):
        P = self.inst.P
        arcs = ({(k, idx): [list(s["seq"]) for s in e.st.sorties[k][idx]]
                 for k in range(P.num_trucks) for idx in range(len(e.st.tG[k])) if e.st.sorties[k][idx]}
                if e.st is not None else {})
        return Sol(P, e.routes, arcs)

    def decode(self, specs, greedy=False):
        envs = [Env(self.pinst, s["lam"], aug=self.next_aug(), route_guide=s.get("route_guide"),
                    arc_guide=s.get("arc_guide"), guide_free=s.get("guide_free", True), blocked=s.get("blocked"))
                for s in specs]
        if envs:
            rollout(self.policy, self.pinst, envs, H=self.enc, greedy=greedy, record=False, chunk=self.chunk)
        return [self.to_sol(e) for e in envs], envs

    # ------------------------------------------------------------------ online
    def replay(self, sols, lams):
        """Quỹ đạo (obs, hành động) tái tạo đúng từng lời giải; bỏ lời giải không tái tạo được."""
        kept = []
        for s, lam in zip(sols, lams):
            e = Env(self.pinst, lam, aug=int(self.rng.integers(self.n_aug)), route_guide=s.routes,
                    arc_guide=s.arcs, guide_free=False, imitate=True)
            ok = True
            while not e.done:
                if e.forced is None:          # lệch khỏi lời giải (mask chặn) -> bỏ
                    ok = False
                    break
                e.step(e.forced, record=True)
            if ok and e.traj and self.to_sol(e).sig == s.sig:
                kept.append(e)
        return kept

    def online_update(self, archive, cfg):
        net, P, rng = self.policy, self.inst.P, self.rng
        if self.opt is None:
            self.opt = torch.optim.Adam(net.adapter_params(), lr=cfg.adapt_lr)
        elite = list(archive.sols)
        if not elite:
            return None, []
        F = np.array([[s.f1, s.f2] for s in elite])
        lam_e = assign_lambda(P, F, F, self.Lam, "weighted")
        sel = np.arange(len(elite))
        if len(sel) > cfg.max_elite:
            sel = rng.choice(len(elite), cfg.max_elite, replace=False)
        imit = self.replay([elite[i] for i in sel], [lam_e[i] for i in sel])

        lams = self.Lam[rng.integers(len(self.Lam), size=cfg.n_lam)]
        envs = [Env(self.pinst, lam, aug=self.next_aug()) for lam in lams for _ in range(cfg.n_traj)]
        rollout(net, self.pinst, envs, H=self.enc, record=True, chunk=self.chunk)
        R = np.array([sum(st["r"] for st in e.traj) for e in envs]).reshape(cfg.n_lam, cfg.n_traj)
        A = R - R.mean(1, keepdims=True)
        A = (A / (A.std() + 1e-8)).reshape(-1)
        steps = [st for e in envs for st in e.traj]
        adv = torch.tensor([a for e, a in zip(envs, A) for _ in e.traj], dtype=torch.float32)
        isteps = [st for e in imit for st in e.traj]

        stats = {"pg": 0.0, "nll": 0.0, "elite": len(imit), "elite_total": len(sel), "n": 0}
        if steps:
            bt, act = collate(steps), torch.tensor([s["act"] for s in steps])
            old = torch.tensor([s["logp"] for s in steps], dtype=torch.float32)
        if isteps:
            ibt, iact = collate(isteps), torch.tensor([s["act"] for s in isteps])
        H = self.enc
        for _ in range(cfg.adapt_epochs):
            perm = torch.from_numpy(rng.permutation(len(steps))) if steps else None
            iperm = torch.from_numpy(rng.permutation(len(isteps))) if isteps else None
            n_mb = max(1, int(np.ceil(max(len(steps), len(isteps)) / cfg.adapt_mb)))
            for m in range(n_mb):
                loss = 0.0
                if steps:
                    idx = perm[m::n_mb]
                    logp, _ = net(H, {k: v[idx] for k, v in bt.items()})
                    lpa = logp.gather(1, act[idx, None]).squeeze(1)
                    ratio = torch.exp(lpa - old[idx])
                    a = adv[idx]
                    pg = -torch.min(ratio * a, ratio.clamp(1 - cfg.adapt_clip, 1 + cfg.adapt_clip) * a).mean()
                    loss = loss + pg
                    stats["pg"] += float(pg.detach())
                if isteps:
                    idx = iperm[m::n_mb]
                    logp, _ = net(H, {k: v[idx] for k, v in ibt.items()})
                    nll = -logp.gather(1, iact[idx, None]).squeeze(1).mean()
                    loss = loss + cfg.alpha * nll
                    stats["nll"] += float(nll.detach())
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.adapter_params(), 1.0)
                self.opt.step()
                stats["n"] += 1
        stats["pg"] /= max(1, stats["n"])
        stats["nll"] /= max(1, stats["n"])
        kids = [self.to_sol(e) for e in envs]
        for s, e in zip(kids, envs):
            s.lam, s.op = e.lam, "O1"
        return stats, kids
