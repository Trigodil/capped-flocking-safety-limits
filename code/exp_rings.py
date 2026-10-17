"""Safe ring equilibria under a common goal: regular n-gons with radial balance h(rho, n) = 0.
Rings with radii more than R apart do not interact."""
import sys

import numpy as np
from scipy.optimize import brentq

import swarm_model as sm


def h(rho, n):
    j = np.arange(1, n)
    u = np.pi * j / n
    c = 2 * rho * np.sin(u)
    nav = min(sm.K_NAV * rho, sm.V_NAV)
    return (sm.pair_force(c) * np.sin(u)).sum() - nav


def ring_root(n):
    lo = sm.R_C * (1 + 1e-9) / (2 * np.sin(np.pi / n))
    hi = sm.D_IN / (2 * np.sin(np.pi / n))
    if h(lo, n) <= 0:
        return None
    return brentq(lambda r: h(r, n), lo, hi, xtol=1e-12)


def constants():
    J = int(np.ceil(sm.R / sm.D_M)) - 1
    assert J * sm.D_IN < sm.R, "hypothesis H fails"
    n0 = 2 * J + 2
    while (J + 1) * sm.R_C * (1 - np.pi**2 * (J + 1) ** 2 / (6 * n0**2)) <= sm.R:
        n0 += 1
    jb = 1
    while jb + 1 <= J and (jb + 1) * sm.R_C * (1 - np.pi**2 * (jb + 1) ** 2 / (6 * n0**2)) < sm.D_IN:
        jb += 1
    sb = sum(j * j for j in range(1, jb + 1))
    st = sum(j * j for j in range(2, J + 1))
    delta = sm.D_M * np.cos(np.pi / n0) / (2 * np.pi * (sb + sm.K * sm.A**2 * st / sm.B))
    dlt = int(np.floor(sm.R / delta)) + 1
    n_lo = max(n0, int(np.ceil(2 * np.pi * sm.V_NAV / (sm.K_NAV * sm.R_C))))
    tp = 2 * sm.P * sum(range(2, J + 1))
    x = np.pi * (2 * sm.C * np.cos(np.pi / n0) - tp) / sm.V_NAV
    n_hi = int(np.ceil(x)) - 1
    return dict(J=J, n0=n0, jb=jb, delta=delta, Delta=dlt, n_lo=n_lo, n_hi=n_hi)


def build(k):
    x = []
    for n in range(k["n_lo"], k["n_hi"] + 1, k["Delta"]):
        rho = ring_root(n)
        a = 2 * np.pi * np.arange(n) / n
        x.append(rho * np.c_[np.cos(a), np.sin(a)])
    return np.vstack(x) if x else np.zeros((0, 2))


def check(x):
    goal = np.zeros(2)
    diff, r = sm.pairwise(x)
    f = sm.pair_force(r)
    f[~np.isfinite(r)] = 0.0
    e = diff / np.where(np.isfinite(r), r, 1.0)[..., None]
    F = sm.sat(sm.K_NAV * (goal - x), sm.V_NAV) + (f[..., None] * e).sum(axis=1)
    return np.abs(F).max(), r.min()


if __name__ == "__main__":
    sm.use(sys.argv[1])
    q = float(sys.argv[2]) if len(sys.argv) > 2 else sm.C / sm.V_NAV
    sm.INSTANCES[sm.NAME]["C"] = q * sm.V_NAV
    sm.use(sm.NAME)
    k = constants()
    print(sm.NAME, f"C/v={q}", k)
    ns = np.arange(k["n_lo"], k["n_hi"] + 1)
    rr = np.array([ring_root(n) for n in ns])
    d = np.diff(rr)
    print(f"ring radii {rr[0]:.3f}..{rr[-1]:.3f}, min step {d.min():.4f} (delta {k['delta']:.4f})")
    x = build(k)
    res, rmin = check(x)
    print(f"N={len(x)} max|F|={res:.2e} min gap={rmin:.4f} r_c={sm.R_C:.4f}")
