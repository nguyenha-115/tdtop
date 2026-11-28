"""Mạng actor-critic cho PPO và hàm chạy song song nhiều môi trường.

    Encoder : Linear(đặc trưng tĩnh) -> TransformerEncoder (pre-norm)          -> H [A, n, d]   (1 lần / instance)
    Ngữ cảnh: [mean H, H[cur], H[a], H[b], MLP(glob) + emb(giai đoạn)]           -> c [B, d]
    Khóa    : H[aug] + W_dyn * dyn (đặc trưng động theo từng nút) ; 2 khóa đặc biệt học được (+ emb giai đoạn)
    Actor   : glimpse MHA(c; khóa) -> con trỏ C*tanh(q.k/sqrt(d)) -> log-softmax có mask
    Critic  : MLP([c, glimpse]) -> V(s)   (dùng chung thân mạng với actor)
    Adapter : g <- g + W2 relu(W1 [g, phi(lambda)])   (lớp thích nghi theo instance kiểu EAS-Lay, có điều kiện
              theo lambda; W2 khởi tạo 0 -> mạng nạp từ checkpoint cũ cho đầu ra y hệt. Khi giải bằng NSGA-II,
              chỉ adapter được tinh chỉnh online - xem rl/ppo/nsga_ctx.py)
"""

import math

import numpy as np
import torch
import torch.nn as nn

from .common import N_AUG, STATIC_DIM
from .env import DYN, GLOB

NEG = -1e9


class PPONet(nn.Module):
    def __init__(self, d=128, layers=3, heads=8, ffn=512, C=10.0):
        super().__init__()
        self.cfg = dict(d=d, layers=layers, heads=heads, ffn=ffn, C=C)
        self.d, self.C = d, C
        self.inp = nn.Linear(STATIC_DIM, d)
        layer = nn.TransformerEncoderLayer(d, heads, ffn, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.enc_norm = nn.LayerNorm(d)
        self.stage = nn.Embedding(2, d)
        self.dyn = nn.Linear(DYN, d)
        self.glob = nn.Sequential(nn.Linear(GLOB, d), nn.ReLU(), nn.Linear(d, d))
        self.ctx = nn.Sequential(nn.Linear(5 * d, d), nn.ReLU(), nn.Linear(d, d))
        self.spec = nn.Parameter(torch.randn(2, d) * 0.1)
        self.glimpse = nn.MultiheadAttention(d, heads, batch_first=True)
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.value = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.adapter = Adapter(d)

    def adapter_params(self):
        return list(self.adapter.parameters())

    def reset_adapter(self):
        self.adapter.reset()

    def encode(self, static):
        """static: tensor [A, n, STATIC_DIM] -> H [A, n, d]."""
        return self.enc_norm(self.enc(self.inp(static)))

    def forward(self, H, bt):
        Ha = H[bt["aug"]]                                                   # [B, n, d]
        B = Ha.shape[0]
        ar = torch.arange(B)
        se = self.stage(bt["stage"])                                        # [B, d]
        keys = torch.cat([Ha + self.dyn(bt["dyn"]), self.spec[None] + se[:, None, :]], 1)   # [B, n+2, d]
        c = self.ctx(torch.cat([Ha.mean(1), Ha[ar, bt["cur"]], Ha[ar, bt["a"]], Ha[ar, bt["b"]],
                                self.glob(bt["glob"]) + se], -1))
        g, _ = self.glimpse(c[:, None, :], keys, keys, key_padding_mask=~bt["mask"], need_weights=False)
        g = g.squeeze(1)
        v = self.value(torch.cat([c, g], -1)).squeeze(-1)                  # critic không qua adapter
        g = self.adapter(g, bt["glob"][:, 10:12])
        u = self.C * torch.tanh((self.wq(g)[:, None, :] * self.wk(keys)).sum(-1) / math.sqrt(self.d))
        logp = torch.log_softmax(u.masked_fill(~bt["mask"], NEG), -1)
        return logp, v

    def save(self, path, extra=None):
        torch.save({"cfg": self.cfg, "state": self.state_dict(), "algo": "ppo", **(extra or {})}, path)

    @staticmethod
    def load(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = PPONet(**ck["cfg"])
        missing, unexpected = net.load_state_dict(ck["state"], strict=False)
        bad = [k for k in missing if not k.startswith("adapter.")] + list(unexpected)
        if bad:
            raise RuntimeError(f"checkpoint không khớp mạng PPO: {bad}")
        return net, ck


class Adapter(nn.Module):
    """Lớp thích nghi residual có điều kiện lambda (FiLM + MLP), đầu ra = 0 lúc khởi tạo."""

    def __init__(self, d, hidden=None):
        super().__init__()
        h = hidden or d
        self.lam = nn.Sequential(nn.Linear(2, h), nn.ReLU())
        self.fc1 = nn.Linear(d, h)
        self.film = nn.Linear(h, 2 * h)
        self.fc2 = nn.Linear(h, d)
        self.reset()

    def reset(self):
        with torch.random.fork_rng(devices=[]):   # không làm lệch chuỗi ngẫu nhiên của lần lấy mẫu sau
            torch.manual_seed(12345)
            for m in (self.lam[0], self.fc1, self.film):
                m.reset_parameters()
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, g, lam):
        z = self.lam(lam)
        gamma, beta = self.film(z).chunk(2, -1)
        h = torch.relu(self.fc1(g) * (1.0 + gamma) + beta)
        return g + self.fc2(h)


# ====================================================================
# Gom quan sát thành tensor
# ====================================================================
def collate(obs):
    t = lambda key, dt: torch.as_tensor(np.stack([o[key] for o in obs]) if isinstance(obs[0][key], np.ndarray)
                                        else np.array([o[key] for o in obs]), dtype=dt)
    return {"stage": t("stage", torch.long), "cur": t("cur", torch.long), "a": t("a", torch.long),
            "b": t("b", torch.long), "aug": t("aug", torch.long), "dyn": t("dyn", torch.float32),
            "glob": t("glob", torch.float32), "mask": t("mask", torch.bool)}


def encode_instance(net, inst, n_aug=N_AUG, grad=False):
    with torch.set_grad_enabled(grad):
        return net.encode(torch.from_numpy(inst.static_aug(n_aug)))


@torch.no_grad()
def rollout(net, inst, envs, H=None, greedy=False, record=True, chunk=512):
    """Chạy đồng thời các env (cùng một instance) tới khi xong."""
    H = encode_instance(net, inst) if H is None else H
    while True:
        live = [e for e in envs if not e.done]
        if not live:
            break
        for s in range(0, len(live), chunk):
            part = live[s:s + chunk]
            logp, v = net(H, collate([e.obs for e in part]))
            act = logp.argmax(-1) if greedy else torch.multinomial(logp.exp(), 1).squeeze(-1)
            lpa = logp.gather(1, act[:, None]).squeeze(1)
            for e, a, lp, vv in zip(part, act.tolist(), lpa.tolist(), v.tolist()):
                e.step(a, lp, vv, record)
    return envs
