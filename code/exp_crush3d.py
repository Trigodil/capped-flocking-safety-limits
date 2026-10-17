"""Common-goal crush sweep in three dimensions, reference instance.
Usage: python exp_crush3d.py <C/v_nav values...>, settles each pack until max|u| < TOL, seeds in parallel."""
import sys
import time
from multiprocessing import Pool

import numpy as np

import swarm_model as sm

GOAL = np.zeros(3)
NS = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
TOL, CHUNK, T_MIN, T_MAX, SEEDS = 1e-3, 50.0, 100.0, 1000.0, 4


def spread_start(n, seed):
    # every gap >= d_in, so any capped pair is made by pressure
    rng = np.random.default_rng(seed)
    half = n ** (1.0 / 3.0) * sm.D_IN
    pts = np.empty((0, 3))
    while len(pts) < n:
        p = rng.uniform(-half, half, size=3)
        if len(pts) == 0 or np.linalg.norm(pts - p, axis=1).min() >= sm.D_IN:
            pts = np.vstack([pts, p])
    return pts


def settle(n, seed):
    h = min(0.005, 0.5 / (sm.K_NAV + 12 * sm.SIGMA))
    x = spread_start(n, seed)
    t, umax = 0.0, np.inf
    while t < T_MAX and (t < T_MIN or umax >= TOL):
        for _ in range(int(CHUNK / h)):
            x = x + h * sm.control(x, GOAL)
        t += CHUNK
        umax = np.linalg.norm(sm.control(x, GOAL), axis=1).max()
    rr = sm.pairwise(x)[1][np.triu_indices(n, 1)]
    return rr.min(), sm.pair_force(rr).max() >= sm.C - 1e-9, umax, t


def use_ratio(q):
    sm.INSTANCES["reference"]["C"] = q * sm.V_NAV
    sm.use("reference")


def job(args):
    use_ratio(args[0])
    return settle(args[1], args[2])


def critical_n(q, pool):
    for n in NS:
        res = pool.map(job, [(q, n, k) for k in range(SEEDS)])
        frac = np.mean([r[1] for r in res])
        print(f"    N={n:3d} capped {frac:.2f} min gap {min(r[0] for r in res):.3f} "
              f"|u|max {max(r[2] for r in res):.1e} T<={max(r[3] for r in res):.0f}", flush=True)
        if frac >= 0.5:
            return n
    return None


if __name__ == "__main__":
    out = []
    pool = Pool(SEEDS)
    for q in map(float, sys.argv[1:]):
        use_ratio(q)
        t0 = time.time()
        print(f"  C/v_nav={q:4.1f} r_c={sm.R_C:.3f}", flush=True)
        n = critical_n(q, pool)
        out.append((q, np.nan if n is None else n))
        print(f"  -> N*={n}  ({time.time() - t0:.0f}s)", flush=True)
        np.save(sm.RESULTS / "crush_reference_3d.npy", np.array(out))
