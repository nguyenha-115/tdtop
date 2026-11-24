"""Hai decoder con trỏ (Pha A / Pha B), FiLM theo lambda (Mục 3.3-3.5) và lớp Policy gộp encoder + decoder."""

import math

import torch
import torch.nn as nn

from .encoder import Encoder
from .environment import NEG
from .utils import N_AUG, TRUCK_DIM

PSI_A, PSI_B, XI_DIM = 4, 8, 7


class FiLM(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 2 * d))
        nn.init.normal_(self.net[-1].weight, std=0.05)            # khác 0: lambda có tác động ngay từ đầu
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, q, lam):
        g, b = self.net((lam - 0.5) * 2.0).chunk(2, -1)
        return (1.0 + g) * q + b


class Glimpse(nn.Module):
    """MHA glimpse (truy vấn 1 vector) + con trỏ C*tanh -> log-softmax có mask."""

    def __init__(self, d, heads, C=10.0):
        super().__init__()
        self.h, self.dk, self.C, self.d = heads, d // heads, C, d
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.pq, self.pk = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)

    def forward(self, q, cand, mask):
        B, M, d = cand.shape
        qh = self.wq(q).view(B, self.h, 1, self.dk)
        kh = self.wk(cand).view(B, M, self.h, self.dk).permute(0, 2, 1, 3)
        vh = self.wv(cand).view(B, M, self.h, self.dk).permute(0, 2, 1, 3)
        s = (qh @ kh.transpose(-1, -2)) / math.sqrt(self.dk)
        s = s.masked_fill(~mask[:, None, None, :], NEG)
        g = self.wo((torch.softmax(s, -1) @ vh).reshape(B, d))
        u = self.C * torch.tanh((self.pq(g)[:, None, :] * self.pk(cand)).sum(-1) / math.sqrt(d))
        return torch.log_softmax(u.masked_fill(~mask, NEG), -1)


class DecoderA(nn.Module):
    def __init__(self, d, heads, C):
        super().__init__()
        self.truck = nn.Sequential(nn.Linear(TRUCK_DIM, d), nn.ReLU(), nn.Linear(d, d))
        self.ctx = nn.Linear(4 * d + PSI_A, d)
        self.film = FiLM(d)
        self.ptr = Glimpse(d, heads, C)
        self.e_end = nn.Parameter(torch.zeros(d))

    def forward(self, enc, bt):
        H, aug, k = enc["H"], bt["aug"], bt["k"]
        htruck = self.truck(enc["xtruck"])[k]
        c = self.ctx(torch.cat([enc["hroad"][aug], enc["hinfo"][aug], H[aug, bt["cur"]], htruck, bt["psi"]], -1))
        c = self.film(c, bt["lam"])
        cand = torch.cat([enc["Hroad"][aug], (enc["Hend"][aug] + self.e_end)[:, None, :]], 1)
        return self.ptr(c, cand, bt["mask"])


class DecoderB(nn.Module):
    def __init__(self, d, heads, C, use_xi=True, max_slots=16):
        super().__init__()
        self.use_xi = use_xi
        self.truck = nn.Sequential(nn.Linear(TRUCK_DIM, d), nn.ReLU(), nn.Linear(d, d))
        self.slot = nn.Embedding(max_slots, d)
        self.ctx = nn.Linear(6 * d + PSI_B, d)
        self.film = FiLM(d)
        self.wxi = nn.Linear(XI_DIM, d, bias=False)
        self.ptr = Glimpse(d, heads, C)
        self.e_stop = nn.Parameter(torch.zeros(d))

    def forward(self, enc, bt):
        H, aug, k = enc["H"], bt["aug"], bt["k"]
        htruck = self.truck(enc["xtruck"])[k]
        c = self.ctx(torch.cat([enc["hinfo"][aug], H[aug, bt["i"]], H[aug, bt["j"]], H[aug, bt["last"]],
                                self.slot(bt["m"]), htruck, bt["psi"]], -1))
        c = self.film(c, bt["lam"])
        xi = bt["xi"] if self.use_xi else torch.zeros_like(bt["xi"])
        cand = enc["Hinfo"][aug] + self.wxi(xi)                                   # k_v = h_v + W_xi xi_v
        stop = (H[aug, bt["j"]] + self.e_stop)[:, None, :]
        return self.ptr(c, torch.cat([cand, stop], 1), bt["mask"])


class Policy(nn.Module):
    def __init__(self, d=128, layers=6, heads=8, ffn=512, C=10.0, edge_bias=True, use_xi=True):
        super().__init__()
        self.cfg = dict(d=d, layers=layers, heads=heads, ffn=ffn, C=C, edge_bias=edge_bias, use_xi=use_xi)
        self.encoder = Encoder(d, layers, heads, ffn, edge_bias)
        self.dec_a = DecoderA(d, heads, C)
        self.dec_b = DecoderB(d, heads, C, use_xi)

    def decoder_params(self):
        """Tham số decoder + FiLM (dùng cho chế độ online: giữ nguyên encoder để dùng cache)."""
        return list(self.dec_a.parameters()) + list(self.dec_b.parameters())

    def encode(self, inst, n_aug=N_AUG, grad=True):
        with torch.set_grad_enabled(grad):
            H = self.encoder(inst.node_tensor(n_aug), inst.types_t, inst.edge_t, inst.glob_t)
        return self.make_enc(inst, H)

    @staticmethod
    def make_enc(inst, H):
        road, info = torch.as_tensor(inst.road), torch.as_tensor(inst.info)
        return {"H": H, "Hroad": H[:, road], "Hinfo": H[:, info], "Hend": H[:, inst.end],
                "hroad": H[:, road].mean(1), "hinfo": H[:, info].mean(1), "xtruck": inst.truck_t}

    def logp(self, enc, bt):
        return (self.dec_a if bt["phase"] == "A" else self.dec_b)(enc, bt)

    @torch.no_grad()
    def act(self, enc, bt, greedy=False):
        lp = self.logp(enc, bt)
        if greedy:
            return lp.argmax(-1)
        return torch.multinomial(lp.exp(), 1).squeeze(-1)

    def save(self, path, extra=None):
        torch.save({"cfg": self.cfg, "state": self.state_dict(), **(extra or {})}, path)

    @staticmethod
    def load(path, **override):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = {**ck["cfg"], **override}
        p = Policy(**cfg)
        p.load_state_dict(ck["state"])
        return p
