"""Giải instance TD-TOP bằng chính sách PPO đã huấn luyện. Hai chế độ, chọn bằng `nsga2` (config) / --nsga2:

nsga2 = false (mặc định): PPO độc lập - thuần RL, không tiến hóa / tìm kiếm cục bộ (mô tả bên dưới).
nsga2 = true            : PPO + NSGA-II - mạng PPO làm toán tử O1-O5 của rl/core/nsga2_kl_rl
    (nsga2_mode: online = tinh chỉnh adapter theo instance + học bắt chước elite; offline = mạng cố định).
    Tham số NSGA-II (pop, gen, ops, rho, ls_steps, ...) lấy từ mục `solve` trong config.yaml; có thể ghi đè
    trên dòng lệnh, vd --gen 50 --ops O1 O2 O3 (tham số lạ được chuyển thẳng cho nsga2_kl_rl).
    Kết quả ghi theo định dạng của nsga2_kl_rl vào `nsga2_out`/<subfolder>/<file>.json.

PPO độc lập, với mỗi instance:
  vòng 0      : giải mã tham lam tại M+1 lambda Das-Dennis x 8 phép D4
  vòng 1..R   : mỗi lambda lấy mẫu thêm `n_samples` lời giải
  sau mỗi vòng: cập nhật kho Pareto (lời giải khả thi không bị trội), ghi HV vào history
Mọi lời giải trên mặt Pareto được kiểm tra lại bằng moo/utils.verify_solution.
Kết quả ghi theo định dạng moo (pareto_front, hypervolume, history, ...) vào moo/result/ppo/<subfolder>/<file>.json
-> so sánh trực tiếp bằng: python moo/compare.py --algo ppo nsga2 nsga3 moead mode

Chạy (trong thư mục rl):
    python ppo/solve.py --subfolder S/N10
    python ppo/solve.py --subfolder L/N100 --filename L042_N100_C_R3.json --rounds 10 --n-samples 16
    python ppo/solve.py --subfolder S/N10 --nsga2                 # PPO + NSGA-II
    python ppo/solve.py --subfolder S/N10 --nsga2 --gen 50 --pop 80
"""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "rl.ppo"

import argparse
import json
import time

import numpy as np
import torch

from .common import DATA_ROOT, N_AUG, RL_DIR, Instance, das_dennis, hv_norm, load_config, moo_utils, rl_path, set_seed
from .env import Env
from .model import PPONet, encode_instance, rollout


class Archive:
    """Kho Pareto theo mục tiêu chuẩn hóa (o1 = 1 - f1/F1_REF, o2 = f2/F2_REF, cùng cực tiểu)."""

    def __init__(self, P):
        self.P, self.items = P, []                     # [(o1, o2, f1, f2, State)]

    def add(self, envs):
        new = list(self.items)
        for e in envs:
            if e.feasible:
                f1, f2 = e.objectives()
                o1, o2 = self.P.normalize(f1, f2)
                new.append((float(o1), float(o2), f1, f2, e.st))
        new.sort(key=lambda t: (t[0], t[1]))
        keep, best = [], np.inf
        for t in new:
            if t[1] < best - 1e-12:
                keep.append(t)
                best = t[1]
        self.items = keep

    def hv(self):
        return hv_norm([(t[0], t[1]) for t in self.items])


def solve_instance(net, path, args, log=print):
    inst = Instance(path)
    P = inst.P
    t0 = time.time()
    H = encode_instance(net, inst)
    Lam = das_dennis(args.M)
    arch = Archive(P)
    hist, evals = {}, 0

    envs = [Env(inst, lam, aug=a) for lam in Lam for a in range(N_AUG)]
    rollout(net, inst, envs, H=H, greedy=True, record=False, chunk=args.chunk)
    arch.add(envs)
    evals += len(envs)
    hist[0] = {"hypervolume": arch.hv(), "time": time.time() - t0, "evaluations": evals, "archive": len(arch.items)}
    log(f"  vòng 0 (greedy): |A|={len(arch.items)} HV={hist[0]['hypervolume']:.4f} t={hist[0]['time']:.1f}s")
    stop = None
    for r in range(1, args.rounds + 1):
        if args.time_limit and time.time() - t0 > args.time_limit:
            stop = "time_limit"
            break
        envs = [Env(inst, lam, aug=int((r * args.n_samples + j) % N_AUG)) for lam in Lam for j in range(args.n_samples)]
        rollout(net, inst, envs, H=H, greedy=False, record=False, chunk=args.chunk)
        arch.add(envs)
        evals += len(envs)
        hist[r] = {"hypervolume": arch.hv(), "time": time.time() - t0, "evaluations": evals,
                   "archive": len(arch.items)}
        if r % args.log_every == 0 or r == args.rounds:
            log(f"  vòng {r}: |A|={len(arch.items)} HV={hist[r]['hypervolume']:.4f} t={hist[r]['time']:.1f}s")
    elapsed = time.time() - t0
    return build_result(P, arch, hist, elapsed, evals, stop, args)


def build_result(P, arch, hist, elapsed, evals, stop, args):
    mu = moo_utils()
    front, all_errs = [], []
    for o1, o2, f1, f2, st in sorted(arch.items, key=lambda t: (t[0], t[1])):
        sol = mu.to_solution(st)
        errs = mu.verify_solution(P, sol)
        all_errs += errs
        front.append({
            "f1_total_value": f1,
            "f2_total_energy_wh": f2,
            "objectives_normalized": [o1, o2],
            "violation_h": sol.violation,
            "verify_errors": errs,
            "routes": [list(r) for r in st.routes],
            "sorties": [{"truck": k, "arc": idx, "nodes": list(s["seq"])} for k, idx, s in st.all_sorties()],
            "solution": mu.solution_to_dict(P, sol, (f1, f2)),
        })
    f1s = [p["f1_total_value"] for p in front]
    f2s = [p["f2_total_energy_wh"] for p in front]
    return {
        "time": elapsed,
        "stop_reason": stop,
        "evaluations": evals,
        "history": hist,
        "hypervolume": arch.hv(),
        "hv_reference": {"f1": 0.0, "f2_wh": P.f2_ref, "f1_norm": P.f1_ref},
        "pareto_size": len(front),
        "ideal_point": {"f1": max(f1s), "f2_wh": min(f2s)} if front else None,
        "nadir_point": {"f1": min(f1s), "f2_wh": max(f2s)} if front else None,
        "num_generations": max(hist) if hist else 0,
        "verify_errors": sorted(set(all_errs)),
        "params": {"algo": "ppo", "ckpt": str(args.ckpt), "M": args.M, "rounds": args.rounds,
                   "n_samples": args.n_samples, "seed": args.seed, "time_limit": args.time_limit},
        "pareto_front": front,
    }


def build_parser():
    p = argparse.ArgumentParser(description="Giải TD-TOP bằng chính sách PPO (mục ppo.solve trong rl/config.yaml)")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--subfolder", type=str, help="vd: S/N10")
    p.add_argument("--filename", type=str, help="vd: S042_N10_R_R3.json")
    p.add_argument("--data-root", type=str)
    p.add_argument("--ckpt", type=str)
    p.add_argument("--M", type=int, help="|Lambda| = M + 1")
    p.add_argument("--rounds", type=int, help="số vòng lấy mẫu sau vòng greedy")
    p.add_argument("--n-samples", type=int, help="số mẫu mỗi lambda mỗi vòng")
    p.add_argument("--time-limit", type=float)
    p.add_argument("--chunk", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--log-every", type=int)
    p.add_argument("--out", type=str)
    p.add_argument("--nsga2", action=argparse.BooleanOptionalAction,
                   help="bật: PPO + NSGA-II (rl/core/nsga2_kl_rl); tắt (--no-nsga2): PPO độc lập")
    p.add_argument("--nsga2-out", type=str, help="thư mục kết quả khi bật --nsga2")
    p.add_argument("--nsga2-mode", choices=["online", "offline"],
                   help="online = tinh chỉnh adapter PPO theo instance trong lúc chạy NSGA-II; offline = mạng cố định")
    p.set_defaults(**{"data_root": str(DATA_ROOT), "ckpt": str(RL_DIR / "outputs/ppo/checkpoints/best.pt"), "M": 100,
                      "rounds": 10, "n_samples": 8, "time_limit": None, "chunk": 1024, "seed": 42, "log_every": 5,
                      "out": str(RL_DIR.parent / "moo/result/ppo"), "nsga2": False,
                      "nsga2_out": str(RL_DIR / "outputs/ppo_nsga2"), "nsga2_mode": "online"})
    cfg_path = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else None
    p.set_defaults(**load_config("solve", cfg_path))
    return p


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args, extra = build_parser().parse_known_args(argv)
    args.ckpt, args.out = rl_path(args.ckpt), rl_path(args.out)
    if args.nsga2:
        return run_nsga2(args, extra)
    if extra:
        raise SystemExit(f"tham số không nhận ra (chỉ dùng được khi bật --nsga2): {' '.join(extra)}")
    set_seed(args.seed)
    torch.set_grad_enabled(False)
    root = Path(args.data_root)
    base = root / args.subfolder if args.subfolder else root
    files = [base / args.filename] if args.filename else sorted(base.glob("*.json"))
    if not files or not files[0].exists():
        raise SystemExit(f"không tìm thấy instance trong {base}")
    net, ck = PPONet.load(args.ckpt)
    net.eval()
    print(f"PPO ckpt: {args.ckpt} (it={ck.get('it')}, val_hv={ck.get('val_hv')})", flush=True)
    for f in files:
        sub = f.parent.relative_to(root).as_posix()
        print(f"=== {sub}/{f.name} ===", flush=True)
        res = solve_instance(net, f, args, log=lambda m: print(m, flush=True))
        out = Path(args.out) / sub / f.name
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False)
        print(f"-> {res['pareto_size']} điểm Pareto, HV={res['hypervolume']:.4f}, {res['time']:.1f}s, "
              f"lỗi verify={len(res['verify_errors'])} | đã lưu {out}", flush=True)


def run_nsga2(args, extra):
    """PPO + NSGA-II: chuyển sang rl/core/nsga2_kl_rl với checkpoint PPO (mode = nsga2_mode)."""
    from rl.core import nsga2_kl_rl
    argv = ["--ckpt", str(args.ckpt), "--mode", args.nsga2_mode, "--data-root", str(args.data_root),
            "--M", str(args.M), "--seed", str(args.seed), "--out", str(rl_path(args.nsga2_out))]
    if args.subfolder:
        argv += ["--subfolder", args.subfolder]
    if args.filename:
        argv += ["--filename", args.filename]
    if args.time_limit:
        argv += ["--time-limit", str(args.time_limit)]
    print(f"PPO + NSGA-II ({args.nsga2_mode}): ckpt={args.ckpt} -> {rl_path(args.nsga2_out)}", flush=True)
    nsga2_kl_rl.main(argv + list(extra))


if __name__ == "__main__":
    main()
