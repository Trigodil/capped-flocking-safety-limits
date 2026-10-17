"""3D crush figures: a settled pack at C/v_nav = 3, N = 192, and measured N* vs C/v_nav with the d = 3 bound.
The pack is the seed-1 run of exp_crush3d.py, cached in pack3d_q3_n192.npy."""
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

import exp_crush3d as e3
import plot_style
import swarm_model as sm

plot_style.apply()
Q, N, SEED, CACHE = 3.0, 192, 1, sm.RESULTS / "pack3d_q3_n192.npy"


def settled_pack():
    if CACHE.exists():
        return np.load(CACHE)
    e3.use_ratio(Q)
    h = min(0.005, 0.5 / (sm.K_NAV + 12 * sm.SIGMA))
    x = e3.spread_start(N, SEED)
    t, umax = 0.0, np.inf
    while t < e3.T_MAX and (t < e3.T_MIN or umax >= e3.TOL):
        for _ in range(int(e3.CHUNK / h)):
            x = x + h * sm.control(x, e3.GOAL)
        t += e3.CHUNK
        umax = np.linalg.norm(sm.control(x, e3.GOAL), axis=1).max()
    np.save(CACHE, x)
    return x


def theorem_bound(qq):
    out = []
    for q in qq:
        e3.use_ratio(q)
        t = sm.D_IN / sm.R_C
        k = math.floor((2 * t + 1) ** 3 - 1)
        out.append((4.0 / 3.0 * (k * q + 1 + 2 * sm.V_NAV / (sm.K_NAV * sm.R_C))) ** 3)
    return np.array(out)


x = settled_pack()
e3.use_ratio(Q)
r = sm.pairwise(x)[1]
dist = np.linalg.norm(x - e3.GOAL, axis=1)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection="3d")
cap = np.triu(r <= sm.R_C + 1e-9, 1)
sc = ax.scatter(*x.T, c=dist, cmap="viridis", s=70, edgecolors="black", linewidths=0.5, alpha=0.9)
for i, j in zip(*np.where(cap)):
    ax.plot(*x[[i, j]].T, color="red", lw=4, zorder=10)
ax.plot([], [], color="red", lw=4, label=r"capped pair, $r\leq r_c$")
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.set_major_locator(MaxNLocator(3))
    axis.pane.fill = False
    axis.pane.set_edgecolor("black")
ax.grid(False)
lim = [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
for k in range(3):
    for a in lim[(k + 1) % 3]:
        for b in lim[(k + 2) % 3]:
            seg = np.empty((2, 3))
            seg[:, k], seg[:, (k + 1) % 3], seg[:, (k + 2) % 3] = lim[k], a, b
            ax.plot(*seg.T, color="black", lw=1)
ax.set_xlim(lim[0])
ax.set_ylim(lim[1])
ax.set_zlim(lim[2])
ax.set_xlabel(r"$x$ (m)", labelpad=12)
ax.set_ylabel(r"$y$ (m)", labelpad=12)
ax.set_zlabel(r"$z$ (m)", labelpad=12)
ax.set_box_aspect((1, 1, 1))
ax.view_init(elev=18, azim=-60)
ax.legend(loc="upper left")
cb = fig.colorbar(sc, ax=ax, shrink=0.45, pad=0.08)
cb.set_label("distance to goal (m)", fontsize=13)
cb.ax.tick_params(labelsize=11)
fig.tight_layout()
fig.savefig(sm.FIGURES / "fig_pack3d.png")
fig.savefig(sm.FIGURES / "fig_pack3d.pdf")

res = np.load(sm.RESULTS / "crush_reference_3d.npy")
q, n = res[:, 0], res[:, 1]
ok = ~np.isnan(n)
slope = np.polyfit(np.log(q[ok][1:]), np.log(n[ok][1:]), 1)[0]
print(f"N*={n.tolist()} slope(q>=1.5)={slope:.2f} capped pairs={int(cap.sum())}")
qq = np.linspace(0.9, 4.4, 200)
fig, bx = plt.subplots(figsize=(8, 6))
bx.loglog(qq, theorem_bound(qq), color=plot_style.BLUE, lw=2, label="bound, $d=3$")
bx.loglog(qq, n[ok][1] * (qq / q[ok][1]) ** 3, "--", color="black", lw=1.5, label=r"slope 3")
bx.loglog(q[ok], n[ok], "o", color=plot_style.BLUE, ms=9, label="simulation")
bx.set_xticks([1, 1.5, 2, 3, 4])
bx.set_xticklabels(["1", "1.5", "2", "3", "4"])
bx.minorticks_off()
bx.set_xlabel(r"$C/v_{nav}$")
bx.set_ylabel(r"$N^*$")
bx.legend()
plot_style.bold_ticks(bx)
fig.tight_layout()
fig.savefig(sm.FIGURES / "fig_crush3d.png")
fig.savefig(sm.FIGURES / "fig_crush3d.pdf")
