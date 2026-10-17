"""All checks on the literature reference instance (OS06 + V14), plus a ratio sweep."""
import numpy as np
from scipy.optimize import minimize

import swarm_model as sm

sm.use("reference")
L = sm.D_IN / 1.2  # length scale relative to the implementation instance
GOAL = np.zeros(2)


def constants():
    print(f"[{sm.NAME}] d_m={sm.D_M:.3f} d_in={sm.D_IN:.3f} d_out={sm.D_OUT:.3f} R={sm.R:.3f}")
    print(f"  b={sm.B:.3f} k={sm.K} C={sm.C} k_n={sm.K_NAV} v_nav={sm.V_NAV} v_max={sm.V_MAX}")
    print(f"  a={sm.A:.3f} r_c={sm.R_C:.4f} eps_w={sm.EPS_W:.4f} p={sm.P:.3f} sigma*={sm.SIGMA:.4f}")
    print(f"  w(r_c)={sm.W_RC:.4f} w(d_m)={sm.pair_potential(np.array([sm.D_M]))[0]:.4f} "
          f"w(0)={sm.pair_potential(np.array([0.0]))[0]:.4f}")
    print(f"  ratios: d_out/d_in={sm.D_OUT/sm.D_IN:.4f} R/d_in={sm.R/sm.D_IN:.4f} d_m/d_in={sm.D_M/sm.D_IN:.4f}")
    r = np.linspace(0.5 * sm.D_M, 1.2 * sm.R, 400001)
    dw = np.gradient(sm.pair_potential(r), r)
    mask = np.abs(r - sm.R) > 1e-2
    print(f"  check max|w'+f| = {np.abs(dw + sm.pair_force(r))[mask].max():.2e}")


def one_sided_rates(x, i, j):
    diff, r = sm.pairwise(x)
    f = sm.pair_force(r)
    f[~np.isfinite(r)] = 0.0
    e = diff / np.where(np.isfinite(r), r, 1.0)[..., None]
    nav = sm.sat(sm.K_NAV * (GOAL - x), sm.V_NAV)
    out = []
    for pull in (-sm.P, 0.0):
        f2 = f.copy()
        f2[i, j] = f2[j, i] = pull
        u = sm.sat(nav + (f2[..., None] * e).sum(axis=1), sm.V_MAX)
        out.append(e[i, j] @ (u[i] - u[j]))
    return out


def cutoff(n=12, h=0.005, T=60.0, seeds=10):
    taus = np.array([0.2, 0.1, 0.05, 0.025])
    near, total, crossings, violations = np.zeros(4), 0, 0, 0
    iu = np.triu_indices(n, 1)
    for s in range(seeds):
        rng = np.random.default_rng(100 + s)
        x = (rng.uniform(-3.0, 3.0, size=(n, 2)) + np.array([5.0, 2.0])) * L
        prev = sm.pairwise(x)[1]
        for _ in range(int(T / h)):
            x = x + h * sm.control(x, GOAL)
            r = sm.pairwise(x)[1]
            dist = np.abs(r[iu] - sm.R)
            near += (dist[None, :] < taus[:, None]).any(axis=1)
            total += 1
            for i, j in np.argwhere(np.triu((prev - sm.R) * (r - sm.R) < 0, 1)):
                rin, rout = one_sided_rates(x, i, j)
                crossings += 1
                violations += rin > rout + 1e-12
            prev = r
    print(f"\n[cutoff] crossings={crossings} violations={violations}")
    for tau, v in zip(taus, near):
        print(f"  tau={tau:.3f} frac={v/total:.5f} frac/tau={v/total/tau:.4f}")


def hessian(x, d=1e-6):
    n = x.size
    H = np.zeros((n, n))
    for k in range(n):
        xp, xm = x.ravel().copy(), x.ravel().copy()
        xp[k] += d
        xm[k] -= d
        H[:, k] = (sm.control(xm.reshape(-1, 2), GOAL).ravel()
                   - sm.control(xp.reshape(-1, 2), GOAL).ravel()) / (2 * d)
    return 0.5 * (H + H.T)


def descent(n=12, T=120.0, seeds=10, hs=(0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.4)):
    print(f"\n[descent] N={n}, common goal, T={T}s, {seeds} seeds")
    iu = np.triu_indices(n, 1)
    lams = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        x = (rng.uniform(-1.8, 1.8, size=(n, 2)) + np.array([6.0, 0.0])) * L
        for _ in range(int(T / 0.005)):
            x = x + 0.005 * sm.control(x, GOAL)
        lams.append((np.linalg.eigvalsh(hessian(x)).max(), sm.pairwise(x)[1][iu].min(),
                     np.linalg.norm(sm.control(x, GOAL), axis=1).max()))
    lams = np.array(lams)
    print(f"  settled: lam_max in [{lams[:,0].min():.3f},{lams[:,0].max():.3f}], "
          f"2/lam in [{2/lams[:,0].max():.3f},{2/lams[:,0].min():.3f}] s, "
          f"min gap in [{lams[:,1].min():.3f},{lams[:,1].max():.3f}], |u|max {lams[:,2].max():.1e}")
    for h in hs:
        inc, late, steps = 0, [], 0
        for s in range(seeds):
            rng = np.random.default_rng(s)
            x = (rng.uniform(-1.8, 1.8, size=(n, 2)) + np.array([6.0, 0.0])) * L
            p = [sm.phi(x, GOAL)]
            for _ in range(int(T / h)):
                x = x + h * sm.control(x, GOAL)
                p.append(sm.phi(x, GOAL))
            d = np.diff(p)
            inc += (d > 1e-9).sum()
            steps += len(d)
            late.append((d[-int(T / 4 / h):] > 1e-9).mean())
        print(f"  h={h:5.3f}s  frac inc={inc/steps:.4f}  late frac inc={np.mean(late):.3f}")


def lattice_energy(s, u, v, reach=4):
    n = np.arange(-reach, reach + 1)
    i, j = np.meshgrid(n, n)
    r = np.hypot(s * (i + j * u), s * j * v).ravel()
    return 0.5 * sm.pair_potential(r[r > 1e-9]).sum()


def best_family(u, v, lo, hi, num=2001):
    ss = np.linspace(lo, hi, num)
    es = np.array([lattice_energy(s, u, v) for s in ss])
    k = es.argmin()
    # flat-bottom: report the whole interval within 1e-9 of the minimum
    flat = ss[es <= es[k] + 1e-9]
    return es[k], flat.min(), flat.max()


def energy_grad(flat):
    x = flat.reshape(-1, 2)
    diff, r = sm.pairwise(x)
    iu = np.triu_indices(len(x), 1)
    e = sm.pair_potential(r[iu]).sum()
    f = sm.pair_force(r)
    f[~np.isfinite(r)] = 0.0
    unit = diff / np.where(np.isfinite(r), r, 1.0)[..., None]
    return e, (-(f[..., None] * unit).sum(axis=1)).ravel()


def psi6(x):
    _, r = sm.pairwise(x)
    vals = []
    for i in range(len(x)):
        nb = np.where(r[i] <= sm.D_OUT + 1e-6)[0]
        if len(nb):
            d = x[nb] - x[i]
            vals.append(abs(np.exp(6j * np.arctan2(d[:, 1], d[:, 0])).mean()))
    return float(np.mean(vals)) if vals else 0.0


def ground_states(nmax=30, restarts=30):
    print("\n[lattices]")
    hexe = best_family(0.5, np.sqrt(3) / 2, sm.D_M, sm.R)
    sqe = best_family(0.0, 1.0, sm.D_M, sm.R)
    print(f"  hex   : e={hexe[0]:.4f}, optimal spacing in [{hexe[1]:.3f},{hexe[2]:.3f}]")
    print(f"  square: e={sqe[0]:.4f}, optimal spacing in [{sqe[1]:.3f},{sqe[2]:.3f}]")
    print(f"  -3 eps_w = {-3*sm.EPS_W:.4f}")
    print(f"\n[ground states] {'N':>3} {'E_N':>9} {'floor':>9} {'(E-eN)/sqrtN':>13} {'min r':>7} {'bonds/N':>8} {'psi6':>5}")
    rng = np.random.default_rng(0)
    eb = hexe[0]
    for n in range(2, nmax + 1):
        best_e, best_x = np.inf, None
        for _ in range(restarts):
            rad = (0.75 * np.sqrt(n) + 0.5) * L
            ang = rng.uniform(0, 2 * np.pi, n)
            rr = rad * np.sqrt(rng.uniform(0, 1, n))
            x0 = np.c_[rr * np.cos(ang), rr * np.sin(ang)]
            res = minimize(energy_grad, x0.ravel(), jac=True, method="L-BFGS-B",
                           options={"maxiter": 5000, "gtol": 1e-10})
            if res.fun < best_e:
                best_e, best_x = res.fun, res.x.reshape(-1, 2)
        r = sm.pairwise(best_x)[1][np.triu_indices(n, 1)]
        bonds = ((r >= sm.D_IN - 1e-6) & (r <= sm.D_OUT + 1e-6)).sum()
        print(f"   {n:3d} {best_e:9.4f} {-sm.EPS_W*n*(n-1)/2:9.4f} {(best_e-eb*n)/np.sqrt(n):13.4f} "
              f"{r.min():7.3f} {bonds/n:8.3f} {psi6(best_x):5.2f}")


def ratio_sweep():
    """Hex lattice optimum vs (d_out/d_in, R/d_in), d_in = 1, d_m = 2/3, k = 1, b = (d_in-d_m)^2."""
    print("\n[ratio sweep] hex optimum spacing / d_in (compressed if < 1)")
    rho1s = [1.1, 1.222, 1.3, 1.414, 1.5]
    rho2s = [1.25, 1.333, 1.5, 1.732, 2.0, 2.083, 2.5]
    print("  d_out/d_in \\ R/d_in " + " ".join(f"{x:7.3f}" for x in rho2s))
    for r1 in rho1s:
        row = []
        for r2 in rho2s:
            if r2 <= r1:
                row.append("    -  ")
                continue
            sm.INSTANCES["sweep"] = dict(D_M=2/3, D_IN=1.0, D_OUT=r1, R=r2, B=(1/3)**2, K=1.0,
                                         C=4.0/6.3, K_NAV=1.0, V_NAV=2.0/6.3, V_MAX=4.0/6.3)
            sm.use("sweep")
            e, lo, hi = best_family(0.5, np.sqrt(3) / 2, 0.7, r2, 1401)
            row.append(f"{lo:7.3f}")
        print(f"  {r1:10.3f}          " + " ".join(row))
    sm.use("reference")


if __name__ == "__main__":
    constants()
    cutoff()
    descent()
    ground_states()
    ratio_sweep()
