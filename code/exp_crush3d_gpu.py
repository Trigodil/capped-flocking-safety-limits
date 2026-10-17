"""GPU version of exp_crush3d.py: all seeds of one pack size settle as one float64 batch in PyTorch.
Usage: python exp_crush3d_gpu.py <C/v_nav> [N values...], with no N values it sweeps exp_crush3d.NS."""
import sys
import time

import numpy as np
import psutil
import torch

import exp_crush3d as e3
import swarm_model as sm

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_FREE_GB = 2.0
if DEV.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.8)


def sat(v, vmax):
    n = torch.linalg.norm(v, dim=-1, keepdim=True)
    return v * torch.clamp(vmax / torch.clamp(n, min=1e-12), max=1.0)


def control(x):
    # x has shape (seeds, n, 3) and the goal is the origin
    diff = x[:, :, None, :] - x[:, None, :, :]
    r = torch.linalg.norm(diff, dim=-1)
    r.diagonal(dim1=1, dim2=2).fill_(float("inf"))
    bar = torch.clamp(sm.B * (1.0 / (r - sm.D_M) - 1.0 / sm.A), max=sm.C)
    f = torch.where(r <= sm.D_M, sm.C, 0.0)
    f = torch.where((r > sm.D_M) & (r < sm.D_IN), bar, f)
    f = torch.where((r > sm.D_OUT) & (r <= sm.R), -sm.K * (r - sm.D_OUT), f)
    sep = ((f / r)[..., None] * diff).sum(dim=2)
    return sat(sat(-sm.K_NAV * x, sm.V_NAV) + sep, sm.V_MAX)


def settle(n):
    h = min(0.005, 0.5 / (sm.K_NAV + 12 * sm.SIGMA))
    x = torch.tensor(np.stack([e3.spread_start(n, k) for k in range(e3.SEEDS)]), device=DEV)
    t = np.zeros(e3.SEEDS)
    umax = np.full(e3.SEEDS, np.inf)
    live = torch.ones(e3.SEEDS, 1, 1, dtype=torch.float64, device=DEV)
    while True:
        go = (t < e3.T_MAX) & ((t < e3.T_MIN) | (umax >= e3.TOL))
        if not go.any():
            break
        live[:, 0, 0] = torch.tensor(go, dtype=torch.float64, device=DEV)
        for _ in range(int(e3.CHUNK / h)):
            x = x + (h * live) * control(x)
        t[go] += e3.CHUNK
        u = torch.linalg.norm(control(x), dim=-1).amax(dim=1).cpu().numpy()
        umax[go] = u[go]
    res = []
    for k, xk in enumerate(x.cpu().numpy()):
        rr = sm.pairwise(xk)[1][np.triu_indices(n, 1)]
        res.append((rr.min(), sm.pair_force(rr).max() >= sm.C - 1e-9, umax[k], t[k]))
    return res


if __name__ == "__main__":
    q = float(sys.argv[1])
    ns = [int(a) for a in sys.argv[2:]] or e3.NS
    e3.use_ratio(q)
    t0 = time.time()
    print(f"  C/v_nav={q:4.1f} r_c={sm.R_C:.3f} on {DEV}", flush=True)
    for n in ns:
        free = psutil.virtual_memory().available / 2**30
        if free < MIN_FREE_GB:
            print(f"    N={n:3d} skipped, {free:.1f} GB RAM free", flush=True)
            break
        res = settle(n)
        frac = np.mean([r[1] for r in res])
        print(f"    N={n:3d} capped {frac:.2f} min gap {min(r[0] for r in res):.3f} "
              f"|u|max {max(r[2] for r in res):.1e} T<={max(r[3] for r in res):.0f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if len(sys.argv) == 2 and frac >= 0.5:
            print(f"  -> N*={n}", flush=True)
            break
