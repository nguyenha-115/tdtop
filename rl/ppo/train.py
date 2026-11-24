"""Huấn luyện PPO độc lập cho TD-TOP (không dùng rl/core).

Mỗi vòng:
  1. Lấy `batch_inst` instance từ tập train (sinh bằng data/data.py, seed >= data_seed, không trùng bộ test 42-46).
  2. Mỗi instance chạy `n_traj` quỹ đạo on-policy, mỗi quỹ đạo một lambda = (w, 1-w), w ~ U(0,1), phép D4 ngẫu nhiên.
  3. GAE(gamma, gae_lambda) trên phần thưởng dày (tổng có trọng số) của env.py; chuẩn hóa lợi thế trên toàn batch.
  4. `epochs` lượt, minibatch `mb` bước (theo từng instance, encoder chạy lại mỗi minibatch):
        L = -mean min(r A, clip(r, 1-eps, 1+eps) A) + vf_coef * mean (V - R)^2 - c_H * mean H(pi)
     dừng sớm khi KL xấp xỉ > target_kl. c_H giảm tuyến tính về 0.
  5. Mỗi `eval_every` vòng: giải mã tham lam trên tập validation cố định -> HV trung bình (theo dõi hội tụ thật).

Chạy (trong thư mục rl):
    python ppo/train.py
    python ppo/train.py --iters 500 --sizes S10 S20 S30 L100 --out outputs/ppo/checkpoints
Hoặc đặt `train.ppo: true` trong config.yaml rồi chạy python core/train.py.
"""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "rl.ppo"

import argparse
import csv
import time

import numpy as np
import torch

from .common import N_AUG, Instance, das_dennis, gen_train_data, hv_norm, load_config, nondominated, rl_path, set_seed
from .env import Env
from .model import PPONet, collate, encode_instance, rollout


# ====================================================================
# Thu thập + GAE
# ====================================================================
def collect(net, inst, n_traj, rng):
    w = rng.uniform(0.0, 1.0, size=n_traj)
    envs = [Env(inst, (wi, 1.0 - wi), aug=int(rng.integers(N_AUG))) for wi in w]
    rollout(net, inst, envs)
    return envs


def gae(envs, gamma, lam):
    """Gắn adv / ret cho từng bước của mỗi quỹ đạo (V(s_T) = 0 ở trạng thái kết thúc)."""
    for e in envs:
        nxt_v, last = 0.0, 0.0
        for s in reversed(e.traj):
            delta = s["r"] + gamma * nxt_v - s["value"]
            last = delta + gamma * lam * last
            s["adv"], s["ret"] = last, last + s["value"]
            nxt_v = s["value"]


def pack(inst, envs):
    steps = [s for e in envs for s in e.traj]
    if not steps:
        return None
    bt = collate(steps)
    f = lambda k: torch.tensor([s[k] for s in steps], dtype=torch.float32)
    return {"inst": inst, "bt": bt, "act": torch.tensor([s["act"] for s in steps]), "old_lp": f("logp"),
            "old_v": f("value"), "adv": f("adv"), "ret": f("ret"), "n": len(steps)}


def sub(bt, idx):
    return {k: v[idx] for k, v in bt.items()}


# ====================================================================
# Cập nhật PPO
# ====================================================================
def ppo_update(net, opt, data, args, c_h, rng):
    st = {"pol": 0.0, "vloss": 0.0, "ent": 0.0, "kl": 0.0, "clip": 0.0, "gnorm": 0.0, "ep": 0}
    n_mb = 0
    for ep in range(args.epochs):
        kl_sum, cnt = 0.0, 0
        for di in rng.permutation(len(data)):
            d = data[di]
            perm = torch.from_numpy(rng.permutation(d["n"]))
            for s in range(0, d["n"], args.mb):
                idx = perm[s:s + args.mb]
                H = encode_instance(net, d["inst"], grad=True)
                logp, v = net(H, sub(d["bt"], idx))
                lpa = logp.gather(1, d["act"][idx, None]).squeeze(1)
                ratio = torch.exp(lpa - d["old_lp"][idx])
                adv = d["adv"][idx]
                pol = -torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv).mean()
                if args.vclip:
                    v_c = d["old_v"][idx] + (v - d["old_v"][idx]).clamp(-args.vclip, args.vclip)
                    vl = torch.max((v - d["ret"][idx]) ** 2, (v_c - d["ret"][idx]) ** 2).mean()
                else:
                    vl = ((v - d["ret"][idx]) ** 2).mean()
                ent = -(logp.exp() * logp).sum(1).mean()
                loss = pol + args.vf_coef * vl - c_h * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip))
                opt.step()
                with torch.no_grad():
                    kl_sum += float((d["old_lp"][idx] - lpa).sum())
                    cnt += len(idx)
                    st["clip"] += float(((ratio - 1).abs() > args.clip).float().mean())
                st["pol"] += float(pol.detach())
                st["vloss"] += float(vl.detach())
                st["ent"] += float(ent.detach())
                st["gnorm"] += gn
                n_mb += 1
        st["ep"] += 1
        st["kl"] = kl_sum / max(1, cnt)
        if args.target_kl and st["kl"] > args.target_kl:
            break
    for k in ("pol", "vloss", "ent", "clip", "gnorm"):
        st[k] /= max(1, n_mb)
    return st


# ====================================================================
# Đánh giá: HV của giải mã tham lam trên tập validation
# ====================================================================
@torch.no_grad()
def evaluate(net, val, M=10):
    hvs = []
    for inst in val:
        H = encode_instance(net, inst)
        envs = [Env(inst, lam, aug=a) for lam in das_dennis(M) for a in range(N_AUG)]
        rollout(net, inst, envs, H=H, greedy=True, record=False)
        P = inst.P
        pts = [P.normalize(*e.objectives()) for e in envs if e.feasible]
        hvs.append(hv_norm(nondominated(pts, key=lambda p: p)))
    return float(np.mean(hvs))


# ====================================================================
# Vòng huấn luyện
# ====================================================================
def build_parser():
    p = argparse.ArgumentParser(description="PPO độc lập cho TD-TOP (mục ppo.train trong rl/config.yaml)")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data-dir", type=str, help="thư mục dữ liệu train/val sinh tự động")
    p.add_argument("--sizes", nargs="+", help="kích thước train, vd: S10 S20 S30 L100")
    p.add_argument("--count", type=int, help="số instance mỗi (kích thước, phân bố)")
    p.add_argument("--data-seed", type=int)
    p.add_argument("--iters", type=int)
    p.add_argument("--batch-inst", type=int)
    p.add_argument("--n-traj", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--epochs", type=int)
    p.add_argument("--mb", type=int, help="số bước mỗi minibatch")
    p.add_argument("--clip", type=float)
    p.add_argument("--vclip", type=float, help="cắt giá trị V quanh V_old (0 = tắt)")
    p.add_argument("--vf-coef", type=float)
    p.add_argument("--ent", type=float, help="hệ số entropy ban đầu (giảm tuyến tính về 0)")
    p.add_argument("--gamma", type=float)
    p.add_argument("--gae-lambda", type=float)
    p.add_argument("--target-kl", type=float)
    p.add_argument("--grad-clip", type=float)
    p.add_argument("--d", type=int)
    p.add_argument("--layers", type=int)
    p.add_argument("--heads", type=int)
    p.add_argument("--ffn", type=int)
    p.add_argument("--init", type=str)
    p.add_argument("--out", type=str)
    p.add_argument("--save-every", type=int)
    p.add_argument("--eval-every", type=int)
    p.add_argument("--log-every", type=int)
    p.add_argument("--seed", type=int)
    p.set_defaults(**{"data_dir": "ppo/data", "sizes": ["S10", "S20", "S30", "L100"], "count": 10,
                      "data_seed": 1000, "iters": 300, "batch_inst": 4, "n_traj": 32, "lr": 1e-4, "epochs": 4,
                      "mb": 512, "clip": 0.2, "vclip": 0.0, "vf_coef": 0.5, "ent": 0.01, "gamma": 1.0,
                      "gae_lambda": 0.95, "target_kl": 0.03, "grad_clip": 1.0, "d": 128, "layers": 3, "heads": 8,
                      "ffn": 512, "init": None, "out": "outputs/ppo/checkpoints", "save_every": 25,
                      "eval_every": 25, "log_every": 1, "seed": 0})
    cfg_path = None
    if "--config" in sys.argv:
        cfg_path = sys.argv[sys.argv.index("--config") + 1]
    p.set_defaults(**load_config("train", cfg_path))
    return p


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data_dir = rl_path(args.data_dir)
    args.out, args.init = rl_path(args.out), rl_path(args.init)
    train_files = gen_train_data(data_dir / "train", args.sizes, count=args.count, seed=args.data_seed)
    val_files = gen_train_data(data_dir / "val", args.sizes, count=1, seed=args.data_seed + 1000)
    print(f"train: {len(train_files)} instance, val: {len(val_files)} instance ({', '.join(args.sizes)})", flush=True)
    cache = {}

    def get(f):
        if f not in cache:
            cache[f] = Instance(f)
        return cache[f]

    val = [get(f) for f in val_files]

    if args.init:
        net, _ = PPONet.load(args.init)
    else:
        net = PPONet(d=args.d, layers=args.layers, heads=args.heads, ffn=args.ffn)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.csv"
    new_log = not log_path.exists() or args.init is None
    log_f = open(log_path, "w" if new_log else "a", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["it", "ret", "f1n", "f2n", "infeasible", "ent", "kl", "clip", "vloss", "gnorm", "val_hv", "t"])
    print(f"PPO: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M tham số -> {out}", flush=True)

    best = -1.0
    t0 = time.time()
    for it in range(1, args.iters + 1):
        c_h = args.ent * max(0.0, 1.0 - (it - 1) / max(1, args.iters))
        data, rets, f1n, f2n, infeas = [], [], [], [], []
        for _ in range(args.batch_inst):
            inst = get(train_files[int(rng.integers(len(train_files)))])
            envs = collect(net, inst, args.n_traj, rng)
            gae(envs, args.gamma, args.gae_lambda)
            d = pack(inst, envs)
            if d is not None:
                data.append(d)
            for e in envs:
                rets.append(sum(s["r"] for s in e.traj))
                infeas.append(not e.feasible)
                if e.feasible:
                    o1, o2 = inst.P.normalize(*e.objectives())
                    f1n.append(1.0 - o1)
                    f2n.append(o2)
        if data:
            allv = torch.cat([d["adv"] for d in data])
            m, s = allv.mean(), allv.std() + 1e-8
            for d in data:
                d["adv"] = (d["adv"] - m) / s
            st = ppo_update(net, opt, data, args, c_h, rng)
        else:
            st = {"pol": 0.0, "vloss": 0.0, "ent": 0.0, "kl": 0.0, "clip": 0.0, "gnorm": 0.0, "ep": 0}
        val_hv = evaluate(net, val) if (args.eval_every and it % args.eval_every == 0) or it == args.iters else None
        row = [it, np.mean(rets), np.mean(f1n) if f1n else 0.0, np.mean(f2n) if f2n else 0.0, np.mean(infeas),
               st["ent"], st["kl"], st["clip"], st["vloss"], st["gnorm"],
               "" if val_hv is None else val_hv, time.time() - t0]
        log.writerow(row)
        log_f.flush()
        if it % args.log_every == 0 or it == 1 or val_hv is not None:
            msg = (f"[PPO] it {it}/{args.iters}  R={row[1]:.4f}  f1n={row[2]:.3f}  f2n={row[3]:.4f}  "
                   f"infeasible={row[4]:.3f}  ent={st['ent']:.3f}  kl={st['kl']:.4f}  clip={st['clip']:.3f}  "
                   f"vloss={st['vloss']:.5f}  gnorm={st['gnorm']:.2f}  ep={st['ep']}  t={row[-1]:.0f}s")
            if val_hv is not None:
                msg += f"  | val_HV={val_hv:.4f}"
            print(msg, flush=True)
        if val_hv is not None and val_hv > best:
            best = val_hv
            net.save(out / "best.pt", {"it": it, "val_hv": val_hv})
        if it % args.save_every == 0 or it == args.iters:
            net.save(out / "last.pt", {"it": it})
    log_f.close()
    print(f"xong: best val_HV={best:.4f}, checkpoint tại {out}", flush=True)
    return net


if __name__ == "__main__":
    main()
