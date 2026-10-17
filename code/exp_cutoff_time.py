"""Cutoff check: time fraction with some |r_ij - R| < tau, and one-sided rates at each crossing.
Zero time on r = R makes the fraction vanish linearly in tau."""
import numpy as np

import swarm_model as sm

N, H, T = 12, 0.001, 20.0
TAUS = [0.04, 0.02, 0.01, 0.005]
SEEDS = range(10)
GOAL = np.zeros(2)


def one_sided_rates(x, i, j):
    """rdot of pair (i,j) with the pair's own tether on (inside) and off (outside)."""
    diff, r = sm.pairwise(x)
    f = sm.pair_force(r)
    f[~np.isfinite(r)] = 0.0
    e = diff / np.where(np.isfinite(r), r, 1.0)[..., None]
    nav = sm.sat(sm.K_NAV * (GOAL - x), sm.V_NAV)
    eij = e[i, j]
    out = []
    for pull in (-sm.K * (sm.R - sm.D_OUT), 0.0):
        f2 = f.copy()
        f2[i, j] = f2[j, i] = pull
        sep = (f2[..., None] * e).sum(axis=1)
        u = sm.sat(nav + sep, sm.V_MAX)
        out.append(eij @ (u[i] - u[j]))
    return out


def main():
    near = np.zeros(len(TAUS))
    total = 0
    crossings, violations = 0, 0
    for s in SEEDS:
        rng = np.random.default_rng(100 + s)
        x = rng.uniform(-3.0, 3.0, size=(N, 2)) + np.array([5.0, 2.0])
        prev = sm.pairwise(x)[1]
        for _ in range(int(T / H)):
            x = x + H * sm.control(x, GOAL)
            r = sm.pairwise(x)[1]
            iu = np.triu_indices(N, 1)
            dist = np.abs(r[iu] - sm.R)
            for k, tau in enumerate(TAUS):
                near[k] += (dist < tau).any()
            total += 1
            crossed = np.argwhere(np.triu((prev - sm.R) * (r - sm.R) < 0, 1))
            for i, j in crossed:
                rin, rout = one_sided_rates(x, i, j)
                crossings += 1
                violations += rin > rout + 1e-12
            prev = r
    print(f"crossings={crossings}, rdot_in > rdot_out violations={violations}")
    print(f"{'tau':>7} {'time frac':>10} {'frac/tau':>9}")
    for tau, n in zip(TAUS, near):
        print(f"{tau:7.3f} {n/total:10.5f} {n/total/tau:9.3f}")


if __name__ == "__main__":
    main()
