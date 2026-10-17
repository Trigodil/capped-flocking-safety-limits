"""Stability of the safe ring equilibria: perturb the rings, run the law, report drift and smallest gap."""
import numpy as np

import exp_rings as er
import swarm_model as sm

for name, q, T in [("reference", 4, 200), ("reference", 10, 200), ("implementation", 10, 60)]:
    sm.use(name)
    sm.INSTANCES[name]["C"] = q * sm.V_NAV
    sm.use(name)
    x0 = er.build(er.constants())
    h = min(0.005, 0.5 / (sm.K_NAV + 12 * sm.SIGMA))
    for amp in [1e-6, 1e-3]:
        x = x0 + amp * np.random.default_rng(0).standard_normal(x0.shape)
        for _ in range(int(T / h)):
            x = x + h * sm.control(x, np.zeros(2))
        r = sm.pairwise(x)[1]
        print(f"{name} C/v={q} N={len(x)} amp={amp:g} min gap {r.min():.4f} r_c {sm.R_C:.4f} "
              f"drift {np.abs(x - x0).max():.4f}", flush=True)
