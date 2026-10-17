"""Common-goal crush sweep: smallest N whose settled packs have a capped pair (r <= r_c).
Usage: python exp_crush.py <instance> <T>, varies C/v_nav with all else fixed."""
import sys
import time

import numpy as np

import swarm_model as sm

GOAL = np.zeros(2)
NS = [4, 6, 8, 10, 12, 15, 19, 24, 30, 38, 48, 60, 75, 95, 120]


def spread_start(n, seed):
    # random start with every gap >= d_in, so any capped pair is made by pressure
    rng = np.random.default_rng(seed)
    half = np.sqrt(n) * sm.D_IN
    pts = []
    while len(pts) < n:
        p = rng.uniform(-half, half, size=2)
        if all(np.hypot(*(p - q)) >= sm.D_IN for q in pts):
            pts.append(p)
    return np.array(pts)


def settle(n, seed, T):
    # step below 2/L of the semiconcavity bound, so no sampled-data chatter
    h = min(0.005, 0.5 / (sm.K_NAV + 12 * sm.SIGMA))
    x = spread_start(n, seed)
    for _ in range(int(T / h)):
        x = x + h * sm.control(x, GOAL)
    _, r = sm.pairwise(x)
    rr = r[np.triu_indices(n, 1)]
    umax = np.linalg.norm(sm.control(x, GOAL), axis=1).max()
    return rr.min(), sm.pair_force(rr).max() >= sm.C - 1e-9, umax


def critical_n(seeds, T):
    for n in NS:
        res = [settle(n, k, T) for k in range(seeds)]
        frac = np.mean([c for _, c, _ in res])
        umax = max(u for _, _, u in res)
        print(f"    N={n:3d} capped {frac:.2f} min gap {min(g for g, _, _ in res):.3f} "
              f"|u|max {umax:.1e}", flush=True)
        if frac >= 0.5:
            return n
    return None


def sweep(name, ratios, seeds=4, T=100.0):
    sm.use(name)
    c0 = sm.C
    out = []
    for q in ratios:
        sm.INSTANCES[name]["C"] = q * sm.V_NAV
        sm.use(name)
        t0 = time.time()
        print(f"  C/v_nav={q:4.1f} r_c={sm.R_C:.3f} sigma*={sm.SIGMA:.1f}", flush=True)
        n = critical_n(seeds, T)
        out.append((q, n))
        print(f"  -> N*={n}  ({time.time() - t0:.0f}s)", flush=True)
    sm.INSTANCES[name]["C"] = c0
    sm.use(name)
    return np.array(out, dtype=float)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "reference"
    T = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
    res = sweep(name, ratios=[1.0, 1.5, 2.0, 3.0, 4.0], T=T)
    np.save(f"crush_{name}_spread.npy", res)
    ok = ~np.isnan(res[:, 1])
    slope = np.polyfit(np.log(res[ok, 0]), np.log(res[ok, 1]), 1)[0]
    print(f"log-log slope N* vs C/v_nav: {slope:.2f}")
