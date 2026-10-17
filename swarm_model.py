"""Idealized Flockwork swarm: pair force f, potential w, potential Phi and control law u.
Instances 'implementation' (deployed defaults) and 'reference' (Olfati-Saber 2006, Viragh et al. 2014).
Call use(name) first; it swaps the module globals in place."""
import numpy as np

INSTANCES = {
    "implementation": dict(
        D_M=0.9, D_IN=1.2, D_OUT=1.8, R=2.5, B=3.0, K=0.5, C=20.0,
        K_NAV=0.6, V_NAV=2.0, V_MAX=2.0),
    "reference": dict(
        D_M=0.6 * 7.0,            # OS06 obstacle spacing ratio d'/d = 0.6
        D_IN=0.9 * 7.0,           # design: flat zone d +- 0.1 d
        D_OUT=1.1 * 7.0,
        R=1.2 * 7.0,              # OS06: d = 7, r = 1.2 d
        B=1.0 * (0.3 * 7.0) ** 2,  # barrier slope at D_IN equals V14 repulsion gain D = 1/s
        K=1.0,                    # V14 linear repulsion gain D = 1/s
        C=4.0,                    # cap = V14 v_max
        K_NAV=1.0,                # V14 target tracking v0 / d = 2 / 2
        V_NAV=2.0,                # V14 v0
        V_MAX=4.0),               # V14 v_max
}


def use(name):
    g = globals()
    g.update(INSTANCES[name])
    g["A"] = D_IN - D_M
    g["R_C"] = D_M + 1.0 / (C / B + 1.0 / A)  # where the barrier hits the cap
    g["EPS_W"] = 0.5 * K * (R - D_OUT) ** 2  # well depth
    g["P"] = K * (R - D_OUT)  # force jump at R
    g["SIGMA"] = B / (R_C - D_M) ** 2  # max upper curvature of w
    g["W_RC"] = -EPS_W + _barrier_integral(R_C)
    g["NAME"] = name


def pair_force(r):
    """Radial force, positive = repulsive."""
    f = np.zeros_like(r)
    f[r <= D_M] = C
    m = (r > D_M) & (r < D_IN)
    f[m] = np.minimum(B * (1.0 / (r[m] - D_M) - 1.0 / A), C)
    m = (r > D_OUT) & (r <= R)
    f[m] = -K * (r[m] - D_OUT)
    return f


def _barrier_integral(r):
    # int_r^{D_IN} B(1/(s-D_M) - 1/A) ds, valid for R_C <= r <= D_IN
    return B * (np.log(A / (r - D_M)) - (D_IN - r) / A)


def pair_potential(r):
    """w(r) = int_r^inf f, so w' = -f and w(inf) = 0."""
    w = np.zeros_like(r)
    m = (r > D_OUT) & (r <= R)
    w[m] = -0.5 * K * ((R - D_OUT) ** 2 - (r[m] - D_OUT) ** 2)
    w[(r >= D_IN) & (r <= D_OUT)] = -EPS_W
    m = (r >= R_C) & (r < D_IN)
    w[m] = -EPS_W + _barrier_integral(r[m])
    m = r < R_C
    w[m] = W_RC + C * (R_C - r[m])
    return w


def nav_potential(x, goal):
    d = np.linalg.norm(x - goal, axis=1)
    rho = V_NAV / K_NAV
    return np.where(d <= rho, 0.5 * K_NAV * d**2, V_NAV * d - V_NAV**2 / (2 * K_NAV))


def sat(v, vmax):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v * np.minimum(1.0, vmax / np.maximum(n, 1e-12))


def pairwise(x):
    diff = x[:, None, :] - x[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(r, np.inf)
    return diff, r


def phi(x, goal):
    _, r = pairwise(x)
    iu = np.triu_indices(len(x), 1)
    return nav_potential(x, goal).sum() + pair_potential(r[iu]).sum()


def control(x, goal, eps=0.0):
    """u_i = D_eps(sat(-grad_i J_i, V_MAX))."""
    diff, r = pairwise(x)
    f = pair_force(r)
    f[~np.isfinite(r)] = 0.0
    e = diff / np.where(np.isfinite(r), r, 1.0)[..., None]
    sep = (f[..., None] * e).sum(axis=1)
    nav = sat(K_NAV * (goal - x), V_NAV)
    u = sat(nav + sep, V_MAX)
    if eps > 0.0:
        u[np.linalg.norm(u, axis=1) < eps] = 0.0
    return u


use("implementation")
