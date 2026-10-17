"""Check that the crush runs are faithful descents: Phi never increases, and the
interaction degree stays small enough for the step to satisfy h < 2/L."""
import sys

import numpy as np

import exp_crush as ec
import swarm_model as sm


def run(name, q, n, seed, T):
    sm.use(name)
    sm.INSTANCES[name]["C"] = q * sm.V_NAV
    sm.use(name)
    h = min(0.005, 0.5 / (sm.K_NAV + 12 * sm.SIGMA))
    x = ec.spread_start(n, seed)
    p0, inc, dmax = sm.phi(x, ec.GOAL), 0, 0
    steps = int(T / h)
    for k in range(steps):
        x = x + h * sm.control(x, ec.GOAL)
        if k % 20 == 0 or k == steps - 1:
            p = sm.phi(x, ec.GOAL)
            inc += p > p0 + 1e-9
            p0 = p
            r = sm.pairwise(x)[1]
            dmax = max(dmax, int((r < sm.R).sum(axis=1).max()))
    L = sm.K_NAV + 2 * sm.SIGMA * dmax
    print(f"{name} C/v={q} N={n} seed={seed}: Phi increases {inc}, max degree {dmax}, "
          f"h={h:.2e}, 2/L={2 / L:.2e}", flush=True)


if __name__ == "__main__":
    run("reference", 4.0, 75, 0, 100.0)
    run("reference", 3.0, 60, 0, 100.0)
    run("implementation", 3.0, 48, 0, 40.0)
    run("implementation", 1.0, 15, 0, 40.0)
