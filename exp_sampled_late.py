"""Implementation instance: fraction of Phi increases over all steps and over the last quarter."""
import numpy as np

import swarm_model as sm

sm.use("implementation")
N, T, GOAL = 12, 30.0, np.zeros(2)
for h in [0.002, 0.005, 0.01, 0.02, 0.04]:
    inc, steps, late = 0, 0, []
    for s in range(20):
        rng = np.random.default_rng(s)
        x = rng.uniform(-1.8, 1.8, size=(N, 2)) + np.array([6.0, 0.0])
        p = [sm.phi(x, GOAL)]
        for _ in range(int(T / h)):
            x = x + h * sm.control(x, GOAL)
            p.append(sm.phi(x, GOAL))
        d = np.diff(p)
        inc += (d > 1e-9).sum()
        steps += len(d)
        late.append((d[-int(T / 4 / h):] > 1e-9).mean())
    print(f"h={h*1e3:4.0f} ms  frac inc={inc/steps:.4f}  late frac inc={np.mean(late):.3f} "
          f"[{min(late):.3f},{max(late):.3f}]", flush=True)
