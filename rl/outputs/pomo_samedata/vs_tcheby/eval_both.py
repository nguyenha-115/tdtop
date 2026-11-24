"""So sánh công bằng policy POMO (rl/core) và PPO (rl/ppo) - thuần policy, không NSGA-II.
Cùng instance test, cùng ngân sách: greedy 101 lambda x 8 D4 + `S` mẫu / lambda; HV chuẩn hóa ref (1,1) như moo."""
import sys, time, json, glob
from pathlib import Path
import numpy as np, torch

ROOT = Path(r"e:/bkai/SVRP/tdtop")
sys.path.insert(0, str(ROOT))
torch.set_grad_enabled(False)

from rl.core.decoder import Policy
from rl.core.environment import Env as CEnv, run_batch
from rl.core.utils import Inst, load_problem
from rl.ppo.common import Instance, das_dennis, hv_norm, nondominated
from rl.ppo.env import Env as PEnv
from rl.ppo.model import PPONet, encode_instance, rollout

core_ck, ppo_ck, S = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8
subs = sys.argv[4].split(",") if len(sys.argv) > 4 else ["S/N10", "S/N20", "S/N30", "L/N100"]
torch.manual_seed(0); np.random.seed(0)
pol = Policy.load(core_ck); pol.eval()
net, _ = PPONet.load(ppo_ck); net.eval()
Lam = das_dennis(100)


def hv_of(P, objs):
    pts = [P.normalize(f1, f2) for f1, f2 in objs]
    nd = nondominated(pts, key=lambda p: p)
    return hv_norm(nd), len(nd), max([o[0] for o in objs], default=0)


def run_core(f):
    P = load_problem(f, 0); inst = Inst(P)
    t0 = time.time(); enc = pol.encode(inst, 8, grad=False)
    envs = [CEnv(inst, l, aug=a) for l in Lam for a in range(8)]
    run_batch(pol, enc, envs, greedy=True)
    objs = [(s.f1, s.f2) for s in (e.solution() for e in envs) if s.feasible]
    g = hv_of(P, objs)
    envs = [CEnv(inst, l, aug=j % 8) for l in Lam for j in range(S)]
    run_batch(pol, enc, envs, greedy=False)
    objs += [(s.f1, s.f2) for s in (e.solution() for e in envs) if s.feasible]
    return g, hv_of(P, objs), time.time() - t0


def run_ppo(f):
    inst = Instance(f); P = inst.P
    t0 = time.time(); H = encode_instance(net, inst)
    envs = [PEnv(inst, l, aug=a) for l in Lam for a in range(8)]
    rollout(net, inst, envs, H=H, greedy=True, record=False, chunk=1024)
    objs = [e.objectives() for e in envs if e.feasible]
    g = hv_of(P, objs)
    envs = [PEnv(inst, l, aug=j % 8) for l in Lam for j in range(S)]
    rollout(net, inst, envs, H=H, greedy=False, record=False, chunk=1024)
    objs += [e.objectives() for e in envs if e.feasible]
    return g, hv_of(P, objs), time.time() - t0


def milp(f):
    m = ROOT / "milp/result/f1" / Path(f).relative_to(ROOT / "data/data")
    return json.load(open(m))["f1_total_value"] if m.exists() else None


rows = []
for sub in subs:
    for f in sorted(glob.glob(str(ROOT / "data/data" / sub / "*.json"))):
        c, p = run_core(f), run_ppo(f)
        rows.append({"sub": sub, "file": Path(f).name, "core_g": c[0][0], "core_s": c[1][0], "core_n": c[1][1], "core_t": c[2],
                     "ppo_g": p[0][0], "ppo_s": p[1][0], "ppo_n": p[1][1], "ppo_t": p[2], "core_f1": c[1][2], "ppo_f1": p[1][2], "milp": milp(f)})
        r = rows[-1]
        print(f"{sub} {r['file']}: POMO g={r['core_g']:.4f} s={r['core_s']:.4f} ({r['core_t']:.1f}s) | "
              f"PPO g={r['ppo_g']:.4f} s={r['ppo_s']:.4f} ({r['ppo_t']:.1f}s)", flush=True)
print("\n=== tổng hợp (HV ref (1,1), trung bình 15 instance) ===")
for sub in subs:
    R = [r for r in rows if r["sub"] == sub]
    m = lambda k: np.mean([r[k] for r in R])
    win = sum(r["ppo_s"] > r["core_s"] + 1e-9 for r in R)
    print(f"{sub:7s} POMO greedy={m('core_g'):.4f} +sample={m('core_s'):.4f} |P|={m('core_n'):.1f} t={m('core_t'):.1f}s"
          f" || PPO greedy={m('ppo_g'):.4f} +sample={m('ppo_s'):.4f} |P|={m('ppo_n'):.1f} t={m('ppo_t'):.1f}s"
          f" || PPO thắng {win}/{len(R)}")
    G = [r for r in R if r["milp"]]
    if G:
        gap = lambda k: 100 * np.mean([(r["milp"] - r[k]) / r["milp"] for r in G])
        print(f"        gap f1 max so với MILP tối ưu: POMO {gap('core_f1'):.1f}%  PPO {gap('ppo_f1'):.1f}%")
json.dump(rows, open(Path(__file__).with_name("eval_rows.json"), "w"), indent=1)
