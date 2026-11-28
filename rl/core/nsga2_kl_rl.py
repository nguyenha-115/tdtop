"""KL-NSGA-II-RL (Mục 6.5): NSGA-II dùng mạng chính sách làm toán tử và được cập nhật lại từ elite.
(Tên giữ "KL" vì KL là cơ sở lý thuyết chọn siêu tham số; cập nhật online chỉ dùng REINFORCE (baseline POMO) + mất mát bắt chước elite, không có số hạng KL.)

Chạy (từ thư mục gốc dự án):
    python -m rl.core.nsga2_kl_rl --subfolder S/N10 --ckpt rl/outputs/checkpoints/last.pt --mode online
    python -m rl.core.nsga2_kl_rl --filename S042_N10_R_R3.json --subfolder S/N10 --ckpt rl/outputs/checkpoints/last.pt --mode offline
Dùng mạng PPO (rl/ppo) làm toán tử (tự nhận ra từ checkpoint, chỉ offline):
    python -m rl.core.nsga2_kl_rl --subfolder S/N10 --ckpt rl/outputs/ppo/checkpoints/best.pt --mode offline
Baseline/ablation:
    --ckpt none --ops O3 O4      NSGA-II thuần (policy khởi tạo ngẫu nhiên, chỉ O3, O4)
    --ops O1 --gen 1             chính sách thuần (chỉ lấy mẫu O1, không tiến hóa)
"""

import sys
from pathlib import Path

if not __package__:  # chạy trực tiếp: python train.py (trong rl/core) -> cho phép import tương đối
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "rl.core"

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from . import operators as ops
from .decoder import Policy
from .environment import Env, run_batch
from .train import build_parser as train_parser, collect, advantages, finalize_args, policy_update
from .utils import (DATA_ROOT, ROOT, Inst, Sol, assign_lambda, build_state, crowding, das_dennis, hv2d,
                    load_problem, load_config, moo_utils, nd_mask, nd_sort, set_seed)

HV_REF = (1.1, 1.1)


# ====================================================================
# Archive không giới hạn (Mệnh đề 6: HV đơn điệu)
# ====================================================================
class Archive:
    def __init__(self):
        self.sols = []

    def objs(self):
        return np.array([s.obj for s in self.sols]) if self.sols else np.zeros((0, 2))

    def would_enter(self, s):
        """s có không bị trội (và không trùng điểm) bởi archive hiện tại không."""
        F = self.objs()
        if len(F) == 0:
            return True
        dominated = ((F <= s.obj).all(1) & (F < s.obj).any(1)).any() or (F == s.obj).all(1).any()
        return not dominated

    def add(self, sols):
        merged = self.sols + [s for s in sols if s.feasible]
        if not merged:
            return
        F = np.array([s.obj for s in merged])
        keep = np.flatnonzero(nd_mask(F))
        seen, out = set(), []
        for i in keep:
            key = (round(F[i, 0], 9), round(F[i, 1], 9))
            if key not in seen:
                seen.add(key)
                out.append(merged[i])
        self.sols = out

    def hv(self):
        return hv2d(self.objs(), HV_REF)


# ====================================================================
# NSGA-II
# ====================================================================
def rank_crowd(pop):
    F = np.array([s.obj for s in pop])
    for r, idx in enumerate(nd_sort(F)):
        cd = crowding(F[idx])
        for i, c in zip(idx, cd):
            pop[i].rank, pop[i].crowd = r, c


def nsga2_select(pool, n):
    F = np.array([s.obj for s in pool])
    out = []
    for idx in nd_sort(F):
        if len(out) + len(idx) <= n:
            out += [pool[i] for i in idx]
        else:
            cd = crowding(F[idx])
            order = np.argsort(-cd, kind="stable")[: n - len(out)]
            out += [pool[idx[i]] for i in order]
            break
        if len(out) == n:
            break
    return out


def tournament(pop, rng):
    a, b = pop[rng.integers(len(pop))], pop[rng.integers(len(pop))]
    return a if (a.rank, -a.crowd) <= (b.rank, -b.crowd) else b


class OperatorSelector:
    """Chọn toán tử thích nghi theo tỉ lệ sinh con vào được archive (có decay)."""

    def __init__(self, names, floor=0.05):
        self.names, self.floor = list(names), floor
        self.succ = {o: 1.0 for o in names}
        self.tot = {o: 2.0 for o in names}

    def probs(self):
        r = np.array([self.succ[o] / self.tot[o] for o in self.names])
        p = r / r.sum()
        p = np.maximum(p, self.floor)
        return p / p.sum()

    def update(self, name, n, ok):
        self.succ[name] += ok
        self.tot[name] += n

    def decay(self, f=0.9):
        for o in self.names:
            self.succ[o] = 1.0 + (self.succ[o] - 1.0) * f
            self.tot[o] = 2.0 + (self.tot[o] - 2.0) * f


# ====================================================================
# Khởi tạo / sinh con
# ====================================================================
def init_population(ctx, n_pop, Lam, max_rounds=6):
    rng = ctx.rng
    pop, sigs = [], set()
    for rd in range(max_rounds):
        need = n_pop - len(pop)
        if need <= 0:
            break
        lams = Lam[rng.integers(len(Lam), size=need)] if rd else Lam[np.linspace(0, len(Lam) - 1, need).astype(int)]
        half = need // 2 if rd == 0 else 0
        sols = ops.o1_sample(ctx, lams[:half], greedy=True) + ops.o1_sample(ctx, lams[half:], greedy=False)
        for s, l in zip(sols, lams):
            if s.feasible and s.sig not in sigs:
                s.lam, s.op = l, "init"
                sigs.add(s.sig)
                pop.append(s)
    return pop


def targeted_lams(ctx, archive, n, Lam):
    """O1 nhắm vào vùng thưa: chọn điểm archive có crowding lớn, dùng lambda của nó (cộng nhiễu nhẹ)."""
    rng = ctx.rng
    if len(archive.sols) < 3:
        return Lam[rng.integers(len(Lam), size=n)]
    cd = crowding(archive.objs())
    cd = np.minimum(cd, 10.0) + 1e-6
    idx = rng.choice(len(archive.sols), size=n, p=cd / cd.sum())
    out = []
    for i in idx:
        lam = archive.sols[i].lam if archive.sols[i].lam is not None else Lam[rng.integers(len(Lam))]
        a = float(np.clip(lam[0] + rng.normal(0, 0.05), 0.0, 1.0))
        out.append((a, 1.0 - a))
    return np.array(out)


def make_children(ctx, pop, archive, counts, args, Lam):
    rng = ctx.rng
    kids = []
    if counts.get("O1", 0):
        n = counts["O1"]
        h = n // 2
        lams = np.concatenate([Lam[rng.integers(len(Lam), size=h)], targeted_lams(ctx, archive, n - h, Lam)])
        for s in ops.o1_sample(ctx, lams):
            s.op = "O1"
            kids.append(s)
    if counts.get("O2", 0):
        par = [tournament(pop, rng) for _ in range(counts["O2"])]
        rho = rng.uniform(*args.rho)
        for s in ops.o2_destroy_rebuild(ctx, par, [p.lam if p.lam is not None else ctx.random_lam() for p in par], rho):
            s.op = "O2"
            kids.append(s)
    if counts.get("O3", 0):
        pairs = [(tournament(pop, rng), tournament(pop, rng)) for _ in range(counts["O3"])]
        lams = [p[0].lam if p[0].lam is not None else ctx.random_lam() for p in pairs]
        for s in ops.o3_route_crossover(ctx, pairs, lams):
            s.op = "O3"
            kids.append(s)
    for _ in range(counts.get("O4", 0)):
        p = tournament(pop, rng)
        lam = p.lam if p.lam is not None else ctx.random_lam()
        s = ops.o4_local_search(ctx, p, lam, args.ls_steps)
        s.op = "O4"
        kids.append(s)
    if args.polish > 0:   # đánh bóng con do mạng sinh ra (O1-O3)
        idx = [i for i, k in enumerate(kids) if k.op in ("O1", "O2", "O3") and rng.random() < args.polish]
        for i, s in zip(idx, ops.polish_sols(ctx, [kids[i] for i in idx])):
            kids[i] = s
    if counts.get("O5", 0):
        n_par = max(1, int(np.ceil(counts["O5"] / (1 + 2 * args.o5_steps))))
        for s in ops.o5_pareto_ladder(ctx, sparse_parents(ctx, archive, pop, n_par), args.o5_steps):
            s.op = "O5"
            kids.append(s)
    return kids


def sparse_parents(ctx, archive, pop, n):
    """Cha cho O5: nửa từ archive theo crowding (vùng thưa của mặt Pareto), nửa theo giải đấu trên quần thể."""
    rng = ctx.rng
    out = [tournament(pop, rng) for _ in range(n - n // 2)]
    if len(archive.sols) >= 3:
        cd = np.minimum(crowding(archive.objs()), 10.0) + 1e-6
        out += [archive.sols[i] for i in rng.choice(len(archive.sols), size=n // 2, p=cd / cd.sum())]
    else:
        out += [tournament(pop, rng) for _ in range(n // 2)]
    return out


# ====================================================================
# Cập nhật chính sách từ elite (Mục 6.4)
# ====================================================================
def online_update(ctx, policy, opt, archive, cfg, Lam, rng):
    inst, P = ctx.inst, ctx.inst.P
    elite = archive.sols
    if not elite:
        return None
    F = np.array([[s.f1, s.f2] for s in elite])
    lam_s = assign_lambda(P, F, F, Lam)
    for s, l in zip(elite, lam_s):
        s.lam = l
    sel = np.arange(len(elite))
    if len(sel) > cfg.max_elite:
        sel = rng.choice(len(elite), cfg.max_elite, replace=False)
    envs = [Env(inst, lam_s[i], route_guide=elite[i].routes, arc_guide=elite[i].arcs, guide_free=False,
                aug=int(rng.integers(cfg.n_aug))) for i in sel]
    run_batch(policy, ctx.enc, envs)
    kept = [e for e, i in zip(envs, sel) if e.solution().sig == elite[i].sig]   # chỉ giữ quỹ đạo tái tạo đúng
    lams = Lam[rng.integers(len(Lam), size=cfg.n_lam)]
    batch = collect(policy, inst, lams, cfg.n_traj, cfg.scalar, cfg.n_aug, enc=ctx.enc)
    batch["adv"] = advantages(batch, True)
    st = policy_update(policy, opt, [batch], cfg, 0.0, elite={"inst": inst, "envs": kept},
                       train_encoder=False, params=policy.decoder_params())
    st["elite_kept"], st["elite_total"] = len(kept), len(envs)
    return st


# ====================================================================
# Thuật toán tổng thể
# ====================================================================
def run(P, policy, args, cfg, log=print):
    rng = np.random.default_rng(args.seed)
    inst = Inst(P)
    Lam = das_dennis(args.M)
    policy.eval()
    t0 = time.time()
    if is_ppo(policy):
        from rl.ppo.nsga_ctx import PPOCtx
        ctx = PPOCtx(inst, policy, rng, Lam, cfg.n_aug, cfg.scalar)
    else:
        enc = policy.encode(inst, cfg.n_aug, grad=False)
        ctx = ops.Ctx(inst, policy, enc, rng, Lam, cfg.n_aug, cfg.scalar)
    online = args.mode == "online"
    opt = torch.optim.Adam(policy.decoder_params(), lr=args.online_lr) if online and not is_ppo(policy) else None

    pop = init_population(ctx, args.pop, Lam)
    if not pop:
        raise RuntimeError("không sinh được lời giải khả thi nào")
    archive = Archive()
    archive.add(pop)
    rank_crowd(pop)
    hist = [(0, archive.hv(), len(archive.sols), time.time() - t0)]
    selector = OperatorSelector(args.ops)
    log(f"gen 0: |P|={len(pop)} |A|={len(archive.sols)} HV={hist[0][1]:.4f}")
    sigs = {s.sig for s in pop} | {s.sig for s in archive.sols}

    for g in range(1, args.gen + 1):
        if args.time_limit and time.time() - t0 > args.time_limit:
            log("hết thời gian")
            break
        # gán lambda cho quần thể theo archive (Mục 6.2) - dùng cho O2/O3/O4
        Fp = np.array([[s.f1, s.f2] for s in pop])
        Fa = np.array([[s.f1, s.f2] for s in archive.sols])
        for s, l in zip(pop, assign_lambda(P, Fp, Fa, Lam, cfg.scalar)):
            s.lam = l
        for s, l in zip(archive.sols, assign_lambda(P, Fa, Fa, Lam, cfg.scalar)):
            s.lam = l
        counts = dict(zip(args.ops, rng.multinomial(args.pop, selector.probs())))
        kids = make_children(ctx, pop, archive, counts, args, Lam)
        msg = ""
        if online and is_ppo(policy) and g % args.U == 0:   # thích nghi adapter PPO (mẫu mới cũng là con)
            with torch.enable_grad():
                st, extra = ctx.online_update(archive, cfg)
            kids += extra
            if st:
                msg = f" | adapt: pg={st['pg']:.3f} nll={st['nll']:.3f} elite={st['elite']}/{st['elite_total']}"
        kids = [k for k in kids if k.feasible]
        uniq = []
        for k in kids:
            if k.sig not in sigs:
                sigs.add(k.sig)
                uniq.append(k)
        # tín dụng toán tử: con vào được archive
        sel_n = {o: 0 for o in args.ops}
        sel_ok = {o: 0 for o in args.ops}
        for k in uniq:
            if k.op not in sel_n:
                continue
            sel_n[k.op] += 1
            sel_ok[k.op] += int(archive.would_enter(k))
        for o in args.ops:
            selector.update(o, sel_n[o], sel_ok[o])
        selector.decay()
        archive.add(uniq)
        pop = nsga2_select(pop + uniq, args.pop)
        rank_crowd(pop)
        hist.append((g, archive.hv(), len(archive.sols), time.time() - t0))
        if online and not is_ppo(policy) and g % args.U == 0:
            st = online_update(ctx, policy, opt, archive, cfg, Lam, rng)
            if st:
                msg = (f" | update: gnorm={st['grad_norm']:.2f} elite={st['elite_kept']}/{st['elite_total']} "
                       f"nll={st['elite_nll']:.2f}")
        if g % args.log_every == 0 or g == args.gen:
            pr = ",".join(f"{o}:{p:.2f}" for o, p in zip(args.ops, selector.probs()))
            log(f"gen {g}: |A|={len(archive.sols)} HV={hist[-1][1]:.4f} new={len(uniq)} ops[{pr}] "
                f"t={time.time() - t0:.1f}s{msg}")
    return archive, hist, time.time() - t0


# ====================================================================
# CLI
# ====================================================================
def build_parser():
    """Mặc định lấy từ rl/config.yaml (mục `solve`); dòng lệnh ghi đè."""
    p = argparse.ArgumentParser(description="KL-NSGA-II-RL cho TD-TOP")
    p.add_argument("--config", type=str, default=None, help="file yaml (mặc định rl/config.yaml)")
    p.add_argument("--filename", type=str, help="vd: S042_N10_R_R3.json")
    p.add_argument("--subfolder", type=str, help="vd: S/N10")
    p.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    p.add_argument("--ckpt", type=str, help="'none' = policy khởi tạo ngẫu nhiên")
    p.add_argument("--mode", choices=["offline", "online"])
    p.add_argument("--pop", type=int)
    p.add_argument("--gen", type=int)
    p.add_argument("--M", type=int, help="Das-Dennis: |Lambda| = M + 1")
    p.add_argument("--ops", nargs="+", choices=list(ops.OPS))
    p.add_argument("--rho", type=float, nargs=2)
    p.add_argument("--ls-steps", type=int)
    p.add_argument("--U", type=int, help="chu kỳ cập nhật mạng (thế hệ)")
    p.add_argument("--online-lr", type=float)
    p.add_argument("--alpha", type=float)
    p.add_argument("--max-elite", type=int)
    p.add_argument("--n-lam", type=int)
    p.add_argument("--n-traj", type=int)
    p.add_argument("--scalar", choices=["tcheby", "weighted"])
    p.add_argument("--no-aug", action="store_true")
    p.add_argument("--polish", type=float, help="xác suất đánh bóng mỗi con của O1-O3 (0 = tắt)")
    p.add_argument("--o5-steps", type=int, help="số bước chèn / xóa của thang Pareto O5")
    p.add_argument("--adapt-lr", type=float, help="PPO online: lr của adapter")
    p.add_argument("--adapt-epochs", type=int)
    p.add_argument("--adapt-mb", type=int)
    p.add_argument("--adapt-clip", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--time-limit", type=float)
    p.add_argument("--log-every", type=int)
    p.add_argument("--out", type=str)
    p.set_defaults(polish=0.0, o5_steps=3, adapt_lr=1e-3, adapt_epochs=4, adapt_mb=512, adapt_clip=0.2)
    p.set_defaults(**load_config("solve"))
    return p


def is_ppo(policy):
    return type(policy).__name__ == "PPONet"


def load_policy(args, cfg):
    """Nạp policy theo checkpoint: mạng POMO (rl/core) hoặc mạng PPO (rl/ppo, nhận ra qua khóa algo='ppo').
    Với PPO: cfg.scalar = "weighted" (PPO học với phần thưởng tổng có trọng số); online = tinh chỉnh adapter
    theo instance (rl/ppo/nsga_ctx.PPOCtx.online_update)."""
    if args.ckpt.lower() == "none":
        return Policy(d=64, layers=2, heads=4, ffn=128)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if ck.get("algo") != "ppo":
        return Policy.load(args.ckpt)
    from rl.ppo.model import PPONet
    net, _ = PPONet.load(args.ckpt)
    if cfg.scalar != "weighted":
        print(f"[PPO] dùng scalar=weighted cho gán lambda / O4 (thay cho {cfg.scalar})", flush=True)
        cfg.scalar = "weighted"
    return net


def make_cfg(args):
    cfg = train_parser().parse_args([])
    for k in ("alpha", "max_elite", "n_lam", "n_traj", "scalar", "no_aug",
              "adapt_lr", "adapt_epochs", "adapt_mb", "adapt_clip"):
        setattr(cfg, k, getattr(args, k))
    cfg.max_elite = args.max_elite
    return finalize_args(cfg)


def collect_files(args):
    root = Path(args.data_root)
    base = root / args.subfolder if args.subfolder else root
    if args.filename:
        return [base / args.filename]
    return sorted(base.rglob("*.json"))


def solution_record(P, s):
    return {"f1": s.f1, "f2": s.f2, "routes": s.routes,
            "sorties": [{"truck": k, "arc": i, "nodes": seq} for (k, i), v in sorted(s.arcs.items()) for seq in v]}


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = make_cfg(args)
    set_seed(args.seed)
    mu = moo_utils()
    files = collect_files(args)
    if not files:
        raise SystemExit("không tìm thấy file instance")
    for f in files:
        P = load_problem(f)
        policy = load_policy(args, cfg)
        print(f"=== {f.name}: |N1|={P.num_road} |V2|={P.num_info} K={P.num_trucks} ===", flush=True)
        archive, hist, el = run(P, policy, args, cfg, log=lambda m: print(m, flush=True))
        errs = 0
        for s in archive.sols:
            errs += len(mu.verify_solution(P, mu.to_solution(build_state(P, s.routes, s.arcs))))
        front = sorted(archive.sols, key=lambda s: s.f1)
        print(f"-> {len(front)} điểm Pareto, HV={archive.hv():.4f}, thời gian {el:.1f}s, lỗi verify={errs}", flush=True)
        res = {"instance": f.name, "mode": args.mode, "ops": args.ops, "time": el, "hv": archive.hv(),
               "verify_errors": errs, "history": [{"gen": g, "hv": h, "archive": a, "t": t} for g, h, a, t in hist],
               "front": [solution_record(P, s) for s in front]}
        out = Path(args.out) / (args.subfolder or "") / f"{f.stem}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False)
        print(f"   đã lưu {out}", flush=True)


if __name__ == "__main__":
    main()
