"""Encoder: Transformer dị thể có bias cạnh (Mục 3.2). Chạy một lần mỗi instance (và mỗi phép D4)."""

import math

import torch
import torch.nn as nn

from .utils import EDGE_DIM, GLOB_DIM, NODE_DIM


class EdgeBiasLayer(nn.Module):
    def __init__(self, d, heads, ffn, de=EDGE_DIM, edge_bias=True):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q, self.k, self.v = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.o = nn.Linear(d, d)
        self.phi = nn.Sequential(nn.Linear(de, 32), nn.ReLU(), nn.Linear(32, heads)) if edge_bias else None
        self.gain = nn.Parameter(torch.tensor(3.0))              # bias cạnh mạnh hơn q.k lúc đầu
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.ReLU(), nn.Linear(ffn, d))

    def forward(self, h, E):
        A, n, d = h.shape
        sp = lambda x: x.view(A, n, self.h, self.dk).transpose(1, 2)
        hn = self.n1(h)
        q, k, v = sp(self.q(hn)), sp(self.k(hn)), sp(self.v(hn))
        s = q @ k.transpose(-1, -2) / math.sqrt(self.dk)
        if self.phi is not None:
            s = s + self.gain * self.phi(E).permute(2, 0, 1)[None]   # a_ij^h += g * phi^h(e_ij)
        out = (torch.softmax(s, -1) @ v).transpose(1, 2).reshape(A, n, d)
        h = h + self.o(out)                                       # pre-norm: giữ nguyên dòng residual
        return h + self.ffn(self.n2(h))


class Encoder(nn.Module):
    def __init__(self, d=128, layers=6, heads=8, ffn=512, edge_bias=True, n_types=4):
        super().__init__()
        self.type_proj = nn.ModuleList([nn.Linear(NODE_DIM, d) for _ in range(n_types)])   # W_type(i)
        self.glob = nn.Sequential(nn.Linear(GLOB_DIM, d), nn.ReLU(), nn.Linear(d, d))
        self.wg = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.wg.weight, std=0.1 / math.sqrt(d))   # thành phần toàn cục nhỏ: không làm mọi nút giống nhau
        self.norm_out = nn.RMSNorm(d)
        self.layers = nn.ModuleList([EdgeBiasLayer(d, heads, ffn, edge_bias=edge_bias) for _ in range(layers)])

    def forward(self, x, types, E, g):
        """x [A,n,NODE_DIM], types [n], E [n,n,EDGE_DIM], g [GLOB_DIM] -> h [A,n,d]."""
        A, n, _ = x.shape
        outs = torch.stack([p(x) for p in self.type_proj], 0)                 # [T,A,n,d]
        idx = types.view(1, 1, n, 1).expand(1, A, n, outs.shape[-1])
        h = outs.gather(0, idx).squeeze(0)
        h = (h - h.mean(1, keepdim=True)) / (h.std(1, keepdim=True) + 1e-3)   # bỏ thành phần chung giữa các nút
        h = h + self.wg(self.glob(g))[None, None, :]
        for layer in self.layers:
            h = layer(h, E)
        return self.norm_out(h)
