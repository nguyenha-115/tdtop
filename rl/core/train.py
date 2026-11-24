"""Pre-train chính sách độc lập (Mục 5.2): policy gradient (REINFORCE) + baseline chia sẻ kiểu POMO, không dùng PPO/critic.

    A^(n) = G(tau^(n)) - mean_nhóm G          (nhóm = cùng instance, cùng lambda; chỉ quỹ đạo khả thi)
    L = -E[ A * sum_t log pi(a_t|s_t, lambda) ] - c_H * H(pi)  [+ alpha * L_elite khi online]
    Mỗi vòng: lấy mẫu on-policy rồi cập nhật đúng 1 bước gradient (không ratio, không clip, không nhiều epoch).

Chạy:  python -m rl.core.train --stages 10 20 50 100 --iters 200 --out rl/outputs/checkpoints
Cập nhật được chia nhỏ theo chunk bước và tích lũy gradient qua "lá" H (encoder chạy 1 lần, backward 1 lần).
"""

import sys
from pathlib import Path

if not __package__:  # chạy trực tiếp: python train.py (trong rl/core) -> cho phép import tương đối
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "rl.core"

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .decoder import Policy
from .environment import Env, run_batch, stack_traj
from .utils import (N_AUG, Inst, load_config, load_problem, build_train_problem, generate_problem, make_train_specs,
                    hv2d, nd_mask, scalarize, set_seed)


# ====================================================================
# Thu thập quỹ đạo
# ====================================================================
def collect(policy, inst, lams, n_traj, scalar="tcheby", n_aug=N_AUG, enc=None, greedy=False):
    """Mỗi (instance, lambda) sinh n_traj quỹ đạo (aug = n mod n_aug). Trả về dict envs/G/valid/group."""
    enc = enc or policy.encode(inst, n_aug, grad=False)
    envs, group = [], []
    for li, lam in enumerate(lams):
        for n in range(n_traj):
            envs.append(Env(inst, lam, aug=n % n_aug))
            group.append(li)
    run_batch(policy, enc, envs, greedy=greedy)
    sols = [e.solution() for e in envs]
    f1 = np.array([s.f1 for s in sols], dtype=float)
    f2 = np.array([s.f2 for s in sols], dtype=float)
    lam_arr = np.array([e.lam for e in envs])
    G = scalarize(inst.P, f1, f2, lam_arr, scalar)
    valid = np.array([s.feasible and not e.repaired for s, e in zip(sols, envs)])
    return {"inst": inst, "envs": envs, "group": np.array(group), "G": G, "valid": valid, "f1": f1, "f2": f2}


def advantages(batch, adv_norm=True):
    """A = G - mean_group(G) (baseline chia sẻ POMO), chỉ trên quỹ đạo khả thi."""
    adv = np.zeros(len(batch["G"]))
    for g in np.unique(batch["group"]):
        idx = np.flatnonzero((batch["group"] == g) & batch["valid"])
        if len(idx) == 0:
            continue
        a = batch["G"][idx] - batch["G"][idx].mean()
        if adv_norm and len(idx) > 1:
            a = a / (a.std() + 1e-8)
        adv[idx] = a
    return adv


# ====================================================================
# Cập nhật
# ====================================================================
def slice_bt(bt, s, e):
    return {k: (v if k == "phase" else v[s:e]) for k, v in bt.items()}


def _parts(inst, envs, adv, n_items, elite_weight=None):
    """Gom bước của các env thành các phần (pha A/B)."""
    parts = []
    tot = sum(len(e.traj) for e in envs)
    for ph in ("A", "B"):
        st = stack_traj(envs, ph)
        if st is None:
            continue
        bt, act, eid = st
        part = {"bt": bt, "act": act, "w": 1.0 / (len(envs) * n_items)}
        if elite_weight is not None:
            part["w"] = elite_weight
        else:
            part["adv"] = torch.as_tensor(adv, dtype=torch.float32)[eid]
            part["w_ent"] = 1.0 / (max(1, tot) * n_items)
        parts.append(part)
    return parts


def _run_parts(policy, inst, Hleaf, parts, cfg, c_h, stats, elite=False):
    for p in parts:
        S = len(p["act"])
        for s in range(0, S, cfg.chunk):
            e = min(S, s + cfg.chunk)
            enc = Policy.make_enc(inst, Hleaf)
            bt = slice_bt(p["bt"], s, e)
            lp = policy.logp(enc, bt)
            lpa = lp.gather(1, p["act"][s:e][:, None]).squeeze(1)
            if elite:
                loss = -lpa.sum() * p["w"]
                stats["elite_nll"] += float(loss.detach())
            else:
                surr = -(p["adv"][s:e] * lpa).sum() * p["w"]            # REINFORCE với baseline POMO
                ent_t = -(lp.exp() * lp.clamp_min(-1e9)).sum(1)
                loss = surr - c_h * ent_t.sum() * p["w_ent"]
                stats["ent"] += float(ent_t.detach().sum()) * p["w_ent"]
                stats["surr"] += float(surr.detach())
            loss.backward()


def policy_update(policy, opt, batches, cfg, c_h, elite=None, train_encoder=True, params=None):
    """batches: list các dict từ collect() (kèm 'adv'); elite: dict(inst, envs) đã replay (hoặc None).
    Thực hiện đúng 1 bước gradient (on-policy). Trả về thống kê để log."""
    n_items = len(batches)
    jobs = [(b["inst"], _parts(b["inst"], [e for e, v in zip(b["envs"], b["valid"]) if v],
                               b["adv"][b["valid"]], n_items), False)
            for b in batches if b["valid"].any()]
    if elite is not None and len(elite["envs"]) and cfg.alpha > 0:
        jobs.append((elite["inst"], _parts(elite["inst"], elite["envs"], None, 1,
                                           elite_weight=cfg.alpha / len(elite["envs"])), True))
    params = params or list(policy.parameters())
    st = {"ent": 0.0, "surr": 0.0, "elite_nll": 0.0}
    opt.zero_grad(set_to_none=True)
    for inst, parts, is_elite in jobs:
        if train_encoder:
            H = policy.encoder(inst.node_tensor(cfg.n_aug), inst.types_t, inst.edge_t, inst.glob_t)
        else:
            with torch.no_grad():
                H = policy.encoder(inst.node_tensor(cfg.n_aug), inst.types_t, inst.edge_t, inst.glob_t)
        Hleaf = H.detach().requires_grad_(True)
        _run_parts(policy, inst, Hleaf, parts, cfg, c_h, st, elite=is_elite)
        if train_encoder and Hleaf.grad is not None:
            H.backward(Hleaf.grad)
    gn = float(torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip))
    opt.step()
    st["grad_norm"] = gn
    return st


# ====================================================================
# Đánh giá nhanh
# ====================================================================
def evaluate_policy(policy, inst, M=10, n_aug=N_AUG, scalar="tcheby"):
    """Giải mã tham lam tại M+1 lambda x n_aug phép D4; trả (HV, số điểm ND, front)."""
    lams = np.array([(i / M, 1 - i / M) for i in range(M + 1)])
    enc = policy.encode(inst, n_aug, grad=False)
    envs = [Env(inst, lam, aug=a) for lam in lams for a in range(n_aug)]
    run_batch(policy, enc, envs, greedy=True)
    sols = [e.solution() for e in envs]
    F = np.array([s.obj for s in sols if s.feasible])
    F = np.unique(F[nd_mask(F)], axis=0) if len(F) else F
    return hv2d(F), len(F), F


# ====================================================================
# Vòng pre-train
# ====================================================================
def build_parser():
    """Mặc định lấy từ rl/config.yaml (mục `train`); dòng lệnh ghi đè. Các cờ ablation để ở đây."""
    p = argparse.ArgumentParser(description="Pre-train policy TD-TOP (REINFORCE + baseline POMO)")
    p.add_argument("--config", type=str, default=None, help="file yaml (mặc định rl/config.yaml)")
    p.add_argument("--stages", type=int, nargs="+", help="curriculum theo số nút thông tin (khi --pool 0)")
    p.add_argument("--iters", type=int, nargs="+", help="số vòng mỗi giai đoạn (1 giá trị = dùng chung)")
    p.add_argument("--batch-inst", type=int, help="số instance mỗi vòng")
    p.add_argument("--n-lam", type=int)
    p.add_argument("--n-traj", type=int, help="N quỹ đạo mỗi (I, lambda) = 8 D4 x N/8")
    p.add_argument("--lr", type=float)
    p.add_argument("--wd", type=float)
    p.add_argument("--ent", type=float, help="c_H ban đầu, giảm tuyến tính về 0")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--scalar", choices=["tcheby", "weighted"], help="ablation 4")
    p.add_argument("--no-aug", action="store_true", help="ablation 8: bỏ tăng cường D4")
    p.add_argument("--no-edge-bias", action="store_true", help="ablation 5")
    p.add_argument("--no-xi", action="store_true", help="ablation 6")
    p.add_argument("--no-adv-norm", action="store_true", help="không chuẩn hóa độ lệch chuẩn của lợi thế")
    p.add_argument("--grad-clip", type=float)
    p.add_argument("--chunk", type=int, help="số bước mỗi chunk khi tính loss")
    p.add_argument("--d", type=int)
    p.add_argument("--layers", type=int)
    p.add_argument("--heads", type=int)
    p.add_argument("--ffn", type=int)
    p.add_argument("--data-dir", type=str, default=None,
                   help="thư mục chứa instance .json (đệ quy), vd ppo/data/train; ưu tiên hơn --pool/--stages")
    p.add_argument("--pool", type=int, help="số instance training set cố định; 0 = curriculum --stages")
    p.add_argument("--pool-seed", type=int)
    p.add_argument("--init", type=str, help="checkpoint khởi tạo")
    p.add_argument("--out", type=str)
    p.add_argument("--save-every", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--ppo", action=argparse.BooleanOptionalAction, default=False,
                   help="huấn luyện mạng PPO độc lập (rl/ppo, tham số ở mục ppo.train) thay cho REINFORCE + POMO")
    p.set_defaults(**load_config("train"))
    return p


def finalize_args(a):
    a.n_aug = 1 if a.no_aug else N_AUG
    return a


def pretrain(args):
    args = finalize_args(args)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.init:
        policy = Policy.load(args.init)
    else:
        policy = Policy(d=args.d, layers=args.layers, heads=args.heads, ffn=args.ffn,
                        edge_bias=not args.no_edge_bias, use_xi=not args.no_xi)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay=args.wd)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files, cache = None, {}
    if args.data_dir:
        d = Path(args.data_dir)
        d = d if d.is_absolute() else Path(__file__).resolve().parents[1] / d   # tương đối -> tính từ rl/
        files = sorted(d.rglob("*.json"))
        if not files:
            raise SystemExit(f"không có instance .json trong {d}")
        print(f"train trên {len(files)} instance từ {d}", flush=True)
    specs = make_train_specs(args.pool, args.pool_seed) if args.pool > 0 and files is None else None
    stages = [0] if (specs or files) else args.stages  # 0 = giai đoạn duy nhất lấy mẫu từ training set

    def sample_inst(n):
        if files:
            f = files[int(rng.integers(len(files)))]
            if f not in cache:
                cache[f] = Inst(load_problem(f, 0))
            return cache[f]
        return Inst(build_train_problem(specs[int(rng.integers(len(specs)))]) if specs else generate_problem(n, rng))

    iters = args.iters * len(stages) if len(args.iters) == 1 else args.iters
    total, done = sum(iters), 0
    t0 = time.time()
    for n, n_it in zip(stages, iters):
        for it in range(1, n_it + 1):
            c_h = args.ent * max(0.0, 1.0 - done / max(1, total))
            batches, infeas = [], []
            for _ in range(args.batch_inst):
                inst = sample_inst(n)
                lams = rng.dirichlet([1.0, 1.0], size=args.n_lam)
                b = collect(policy, inst, lams, args.n_traj, args.scalar, args.n_aug)
                b["adv"] = advantages(b, not args.no_adv_norm)
                batches.append(b)
                infeas.append(1.0 - b["valid"].mean())
            st = policy_update(policy, opt, batches, args, c_h)
            done += 1
            if it % 5 == 0 or it == 1:
                G = np.mean([b["G"][b["valid"]].mean() for b in batches if b["valid"].any()])
                print(f"[N{n}] it {it}/{n_it}  G={G:.4f}  ent={st['ent']:.3f}  "
                      f"gnorm={st['grad_norm']:.2f}  infeasible={np.mean(infeas):.3f}  t={time.time() - t0:.0f}s", flush=True)
            if it % args.save_every == 0:
                policy.save(out / "last.pt", {"stage": n, "it": it})
        ev = (build_train_problem(make_train_specs(1, 1234, (100, 100))[0]) if (specs or files)
              else generate_problem(n, np.random.default_rng(1234)))
        hv, npts, _ = evaluate_policy(policy, Inst(ev), n_aug=args.n_aug)
        print(f"== xong giai đoạn N{n}: HV(greedy)={hv:.4f}, điểm ND={npts}", flush=True)
        policy.save(out / f"policy_N{n}.pt", {"stage": n})
        policy.save(out / "last.pt", {"stage": n})
    return policy


if __name__ == "__main__":
    _known, _rest = build_parser().parse_known_args()
    if _known.ppo:
        # PPO độc lập: chuyển toàn bộ tham số dòng lệnh (trừ --ppo) cho rl/ppo/train.py
        from rl.ppo.train import main as ppo_main
        ppo_main([a for a in sys.argv[1:] if a not in ("--ppo", "--no-ppo")])
    else:
        pretrain(build_parser().parse_args())
