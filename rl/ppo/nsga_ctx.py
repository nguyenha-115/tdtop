"""Cắm chính sách PPO vào KL-NSGA-II-RL (rl/core/nsga2_kl_rl.py) ở chế độ offline.

PPOCtx thay rl/core/operators.Ctx: cùng giao diện decode(specs, greedy) -> (list Sol, list Env), nên các toán tử
O1 (lấy mẫu), O2 (phá-xây), O3 (lai ghép lộ trình), O4 (tìm kiếm cục bộ) dùng nguyên như với mạng POMO.
Teacher forcing của O2/O3 do rl/ppo/env.Env đảm nhận (route_guide, arc_guide, guide_free, blocked).
Lời giải được dựng lại bằng rl/core/utils.Sol (simulator moo là trọng tài cuối cùng) như với POMO.
"""

from rl.core.operators import Ctx
from rl.core.utils import Sol

from .common import Instance
from .env import Env
from .model import encode_instance, rollout


class PPOCtx(Ctx):
    def __init__(self, inst, net, rng, Lam, n_aug=8, scalar="weighted", chunk=1024):
        self.pinst = Instance(inst.P.name)
        H = encode_instance(net, self.pinst, n_aug)
        super().__init__(inst, net, H, rng, Lam, n_aug, scalar)
        self.chunk = chunk

    def decode(self, specs, greedy=False):
        envs = [Env(self.pinst, s["lam"], aug=self.next_aug(), route_guide=s.get("route_guide"),
                    arc_guide=s.get("arc_guide"), guide_free=s.get("guide_free", True), blocked=s.get("blocked"))
                for s in specs]
        if envs:
            rollout(self.policy, self.pinst, envs, H=self.enc, greedy=greedy, record=False, chunk=self.chunk)
        P = self.inst.P
        sols = [Sol(P, e.routes, {(k, idx): [list(s["seq"]) for s in e.st.sorties[k][idx]]
                                  for k in range(P.num_trucks) for idx in range(len(e.st.tG[k]))
                                  if e.st.sorties[k][idx]} if e.st is not None else {})
                for e in envs]
        return sols, envs
