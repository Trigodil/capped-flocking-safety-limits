"""Rerun crush cases that had not settled, until max|u| < TOL or T_MAX.
Same seeds and starts as exp_crush.py, so each run continues the same trajectory."""
import sys

import numpy as np

import exp_crush as ec
import swarm_model as sm

TOL, CHUNK, T_MAX = 1e-3, 50.0, 2000.0


def settle_until(n, seed):
    h = min(0.005, 0.5 / (sm.K_NAV + 12 * sm.SIGMA))
    x = ec.spread_start(n, seed)
    t = 0.0
    while True:
        for _ in range(int(CHUNK / h)):
            x = x + h * sm.control(x, ec.GOAL)
        t += CHUNK
        umax = np.linalg.norm(sm.control(x, ec.GOAL), axis=1).max()
        if umax < TOL or t >= T_MAX:
            break
    rr = sm.pairwise(x)[1][np.triu_indices(n, 1)]
    return t, umax, rr.min(), sm.pair_force(rr).max() >= sm.C - 1e-9


if __name__ == "__main__":
    name = sys.argv[1]
    cases = [tuple(map(float, c.split(":"))) for c in sys.argv[2:]]
    sm.use(name)
    for q, n in cases:
        sm.INSTANCES[name]["C"] = q * sm.V_NAV
        sm.use(name)
        res = [settle_until(int(n), k) for k in range(4)]
        frac = np.mean([c for *_, c in res])
        print(f"C/v={q} N={int(n)} capped {frac:.2f} | " + " ".join(
            f"[T={t:.0f} u={u:.1e} gap={g:.2f} cap={int(c)}]" for t, u, g, c in res), flush=True)
