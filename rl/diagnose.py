"""Chẩn đoán mạng: encoder (mã hóa), dòng gradient, độ nhạy theo lambda, và thời gian chạy.

Chạy:  python -m rl.diagnose --ckpt rl/outputs/checkpoints/last.pt            (từ thư mục gốc)
       python diagnose.py --n 100 --traj 8                                          (trong rl)
So sánh với mạng khởi tạo ngẫu nhiên cùng cấu hình (mặc định bật; tắt bằng --no-init).
"""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "rl"

import argparse
import math
import time

import numpy as np
import torch

from .core.decoder import Policy
from .core.environment import Env, run_batch, stack_traj
from .core.train import _parts, _run_parts, advantages, build_parser, collect, finalize_args
from .core.utils import N_AUG, ROOT, Inst, build_train_problem, make_train_specs, set_seed


# ====================================================================
# 1. Encoder
# ====================================================================
@torch.no_grad()
def encoder_report(policy, inst):
    enc = policy.encoder
    x, types, E, g = inst.node_tensor(N_AUG), inst.types_t, inst.edge_t, inst.glob_t
    A, n, _ = x.shape
    outs = torch.stack([p(x) for p in enc.type_proj], 0)
    idx = types.view(1, 1, n, 1).expand(1, A, n, outs.shape[-1])
    h = outs.gather(0, idx).squeeze(0)
    h = (h - h.mean(1, keepdim=True)) / (h.std(1, keepdim=True) + 1e-3)
    h = h + enc.wg(enc.glob(g))[None, None, :]
    info = torch.as_tensor(inst.info)
    rows = []

    def stats(h):
        z = h[0]                                                   # 1 phép D4, [n, d]
        zi = z[info]
        std_node = float(zi.std(0).mean())                         # độ đa dạng giữa các nút (theo từng chiều)
        zn = torch.nn.functional.normalize(zi - zi.mean(0, keepdim=True) * 0, dim=-1)
        cos = zn @ zn.T
        m = zi.shape[0]
        cos_mean = float((cos.sum() - m) / max(1, m * (m - 1)))    # ~1 => mọi nút giống nhau (over-smoothing)
        s = torch.linalg.svdvals(zi - zi.mean(0, keepdim=True))
        p = (s ** 2) / (s ** 2).sum().clamp_min(1e-12)
        eff_rank = float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))   # số chiều hữu dụng
        return std_node, cos_mean, eff_rank, float(z.norm(dim=-1).mean())

    rows.append(("embedding vào", *stats(h), None, None))
    for li, L in enumerate(enc.layers):
        sp = lambda t: t.view(A, n, L.h, L.dk).transpose(1, 2)
        hn = L.n1(h)
        q, k = sp(L.q(hn)), sp(L.k(hn))
        s = q @ k.transpose(-1, -2) / math.sqrt(L.dk)              # [A,H,n,n]
        bias = L.gain * L.phi(E).permute(2, 0, 1)[None] if L.phi is not None else torch.zeros_like(s[:1])
        att = torch.softmax(s + bias, -1)
        ent = float(-(att * att.clamp_min(1e-12).log()).sum(-1).mean() / math.log(n))   # 1 = đều, 0 = nhọn
        ratio = float(bias.std() / s.std().clamp_min(1e-9))        # bias cạnh mạnh/yếu so với q.k
        h = L(h, E)
        rows.append((f"sau lớp {li + 1}", *stats(h), ent, ratio))
    return rows


def print_encoder(rows):
    print(f"  {'':<14}{'std nút':>9}{'cos TB':>9}{'rank hh':>9}{'|h|':>8}{'H attn':>9}{'bias/qk':>9}")
    for r in rows:
        f = lambda v, w, p: (f"{v:>{w}.{p}f}" if v is not None else " " * w)
        print(f"  {r[0]:<14}{f(r[1], 9, 3)}{f(r[2], 9, 3)}{f(r[3], 9, 2)}{f(r[4], 8, 2)}{f(r[5], 9, 3)}{f(r[6], 9, 3)}")
    print("  (cos TB gần 1 hoặc rank hh rất nhỏ => các nút bị 'trộn' thành một; H attn gần 1 => attention gần như đều;"
          " bias/qk nhỏ => bias cạnh bị bỏ qua)")


# ====================================================================
# 2. Gradient
# ====================================================================
def grad_report(policy, inst, cfg, n_lam, n_traj):
    rng = np.random.default_rng(0)
    lams = rng.dirichlet([1.0, 1.0], size=n_lam)
    b = collect(policy, inst, lams, n_traj, cfg.scalar, cfg.n_aug)
    b["adv"] = advantages(b, True)
    ok = b["valid"]
    if not ok.any():
        return None, b
    envs = [e for e, v in zip(b["envs"], ok) if v]
    parts = _parts(inst, envs, b["adv"][ok], 1)
    policy.zero_grad(set_to_none=True)
    H = policy.encoder(inst.node_tensor(cfg.n_aug), inst.types_t, inst.edge_t, inst.glob_t)
    Hleaf = H.detach().requires_grad_(True)
    st = {"ent": 0.0, "surr": 0.0, "elite_nll": 0.0}
    _run_parts(policy, inst, Hleaf, parts, cfg, 0.0, st)
    H.backward(Hleaf.grad)
    groups = {"encoder.type_proj/glob": [], **{f"encoder.layer{i + 1}": [] for i in range(len(policy.encoder.layers))},
              "dec_a": [], "dec_b": []}
    for name, p in policy.named_parameters():
        if p.grad is None:
            continue
        if name.startswith("encoder.layers."):
            groups[f"encoder.layer{int(name.split('.')[2]) + 1}"].append(p)
        elif name.startswith("encoder."):
            groups["encoder.type_proj/glob"].append(p)
        else:
            groups["dec_a" if name.startswith("dec_a") else "dec_b"].append(p)
    out = {}
    for k, ps in groups.items():
        if ps:
            g = torch.sqrt(sum((p.grad ** 2).sum() for p in ps))
            w = torch.sqrt(sum((p.detach() ** 2).sum() for p in ps))
            out[k] = (float(g), float(w))
    out["_hleaf_grad"] = (float(Hleaf.grad.norm()), 0.0)
    policy.zero_grad(set_to_none=True)
    return out, b


# ====================================================================
# 3. Quyết định của policy
# ====================================================================
@torch.no_grad()
def policy_report(policy, inst, batch):
    envs = batch["envs"]
    enc = policy.encode(inst, N_AUG, grad=False)
    res = {}
    for ph in ("A", "B"):
        st = stack_traj(envs, ph)
        if st is None:
            continue
        bt, act, _ = st
        lp = policy.logp(enc, bt)
        p = lp.exp()
        ent = -(p * lp.clamp_min(-1e9)).sum(1)
        nvalid = bt["mask"].sum(1).float()
        ent_max = nvalid.log()                                      # entropy nếu đều trên tập hợp lệ
        top = p.max(1).values
        # độ nhạy theo lambda: đổi lambda sang hai cực
        l0, l1 = bt["lam"].clone(), bt["lam"].clone()
        l0[:] = torch.tensor([1.0, 0.0])
        l1[:] = torch.tensor([0.0, 1.0])
        lp0 = policy.logp(enc, {**bt, "lam": l0})
        lp1 = policy.logp(enc, {**bt, "lam": l1})
        kl01 = (lp0.exp() * (lp0 - lp1)).sum(1)
        diff_arg = (lp0.argmax(1) != lp1.argmax(1)).float().mean()
        res[ph] = dict(steps=len(act), n_valid=float(nvalid.mean()), ent=float(ent.mean()),
                       ent_ratio=float((ent / ent_max.clamp_min(1e-9))[nvalid > 1].mean()), top=float(top.mean()),
                       kl_lam=float(kl01.mean()), arg_diff=float(diff_arg))
    # độ lớn đóng góp xi trong decoder B so với embedding
    xi_scale = None
    stb = stack_traj(envs, "B")
    if stb is not None:
        bt = stb[0]
        wxi = policy.dec_b.wxi(bt["xi"][:2000])
        h = enc["Hinfo"][bt["aug"][:2000]]
        xi_scale = float(wxi.norm(dim=-1).mean() / h.norm(dim=-1).mean())
    film = [float(m.net[-1].weight.norm()) for m in (policy.dec_a.film, policy.dec_b.film)]
    return res, xi_scale, film


# ====================================================================
# 4. Thời gian
# ====================================================================
def timing_report(policy, inst, n_env):
    t0 = time.time()
    enc = policy.encode(inst, N_AUG, grad=False)
    t_enc = time.time() - t0
    acts = {"t": 0.0, "n": 0}
    orig = policy.act

    def timed(*a, **k):
        s = time.time()
        r = orig(*a, **k)
        acts["t"] += time.time() - s
        acts["n"] += 1
        return r
    policy.act = timed
    envs = [Env(inst, np.array([0.5, 0.5]), aug=i % N_AUG) for i in range(n_env)]
    t0 = time.time()
    run_batch(policy, enc, envs)
    t_run = time.time() - t0
    policy.act = orig
    steps = sum(len(e.traj) for e in envs)
    return dict(t_enc=t_enc, t_run=t_run, t_net=acts["t"], calls=acts["n"], steps=steps,
                n_env=n_env)


# ====================================================================
def diagnose(policy, inst, cfg, label, args):
    policy.eval()
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    print("[1] ENCODER (mã hóa) - 1 instance, phép D4 đầu tiên")
    print_encoder(encoder_report(policy, inst))

    print("\n[2] GRADIENT sau 1 lần tính loss REINFORCE (|grad| / |trọng số| - nhỏ => lớp đó gần như không học)")
    g, batch = grad_report(policy, inst, cfg, args.lam, args.traj)
    if g is None:
        print("  không có quỹ đạo khả thi để tính")
        return
    for k, (gn, wn) in g.items():
        if k == "_hleaf_grad":
            print(f"  |dL/dH| (gradient tại đầu ra encoder) = {gn:.3e}")
        else:
            print(f"  {k:<26} |g|={gn:.3e}  |w|={wn:7.2f}  |g|/|w|={gn / max(wn, 1e-12):.2e}")
    first, last = [v for k, v in g.items() if k == "encoder.layer1"][0], g["encoder.layer%d" % len(policy.encoder.layers)]
    print(f"  tỉ lệ gradient lớp 1 / lớp cuối của encoder = {first[0] / max(last[0], 1e-12):.3f}"
          " (rất nhỏ => gradient bị tắt khi đi ngược về đầu mạng)")

    print("\n[3] QUYẾT ĐỊNH CỦA POLICY (trên các bước đã lấy mẫu)")
    res, xi_scale, film = policy_report(policy, inst, batch)
    for ph, r in res.items():
        print(f"  Pha {ph}: {r['steps']:>6} bước, ứng viên hợp lệ TB={r['n_valid']:6.1f}, entropy TB={r['ent']:.3f} "
              f"({r['ent_ratio'] * 100:.0f}% mức đều), xác suất top-1 TB={r['top']:.3f}")
        print(f"          độ nhạy lambda: KL(lam=(1,0) || lam=(0,1)) = {r['kl_lam']:.4f}, "
              f"đổi top-1 ở {r['arg_diff'] * 100:.1f}% bước")
    if xi_scale is not None:
        print(f"  |W_xi * xi| / |h| (đóng góp đặc trưng động vào ứng viên Pha B) = {xi_scale:.3f}")
    print(f"  |trọng số lớp cuối FiLM| A={film[0]:.3f}, B={film[1]:.3f} (khởi tạo ~0.05*sqrt(d); KL theo lambda ~0 => lambda chưa được dùng)")
    print(f"  khả thi: {batch['valid'].mean() * 100:.0f}%   G TB={batch['G'][batch['valid']].mean():.4f}   "
          f"độ lệch chuẩn G giữa quỹ đạo cùng nhóm TB={np.mean([batch['G'][(batch['group'] == k) & batch['valid']].std() for k in np.unique(batch['group'])]):.4f}")

    print("\n[4] THỜI GIAN")
    t = timing_report(policy, inst, args.traj * args.lam)
    env_t = t["t_run"] - t["t_net"]
    print(f"  encoder: {t['t_enc']:.2f}s (1 lần / instance)")
    print(f"  giải mã {t['n_env']} lời giải: {t['t_run']:.2f}s = mạng {t['t_net']:.2f}s ({t['t_net'] / t['t_run'] * 100:.0f}%) "
          f"+ môi trường Python {env_t:.2f}s ({env_t / t['t_run'] * 100:.0f}%); {t['steps']} bước quyết định, "
          f"{t['calls']} lần gọi mạng")
    print(f"  ~{t['t_run'] / t['n_env'] * 1000:.0f} ms / lời giải")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "rl" / "outputs" / "checkpoints" / "last.pt"))
    ap.add_argument("--n", type=int, default=100, help="số nút thông tin của instance chẩn đoán")
    ap.add_argument("--lam", type=int, default=2, help="số lambda")
    ap.add_argument("--traj", type=int, default=8, help="số quỹ đạo mỗi lambda")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-init", action="store_true", help="không so sánh với mạng khởi tạo ngẫu nhiên")
    args = ap.parse_args()
    set_seed(args.seed)
    cfg = finalize_args(build_parser().parse_args([]))
    spec = make_train_specs(1, args.seed, (args.n, args.n))[0]
    inst = Inst(build_train_problem(spec))
    print(f"instance: {inst.num_info} nút thông tin, {inst.num_road} nút đường bộ, K={inst.K}, R={inst.R[0]}")
    if args.ckpt.lower() == "none":
        args.no_init = True
        policy = Policy(d=cfg.d, layers=cfg.layers, heads=cfg.heads, ffn=cfg.ffn)
        diagnose(policy, inst, cfg, "MẠNG KHỞI TẠO NGẪU NHIÊN (cấu hình hiện tại)", args)
        return
    policy = Policy.load(args.ckpt)
    diagnose(policy, inst, cfg, f"MẠNG ĐÃ TRAIN: {args.ckpt}", args)
    if not args.no_init:
        set_seed(args.seed)
        diagnose(Policy(**policy.cfg), inst, cfg, "MẠNG KHỞI TẠO NGẪU NHIÊN (cùng cấu hình) - để so sánh", args)


if __name__ == "__main__":
    main()
