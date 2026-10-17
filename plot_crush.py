"""Crush figure: measured N* vs C/v_nav for both instances, with the Theorem bounds."""
import math

import matplotlib.pyplot as plt
import numpy as np

import exp_rings as er
import plot_style
import swarm_model as sm

plot_style.apply()


def theorem_bound(name, qq):
    sm.use(name)
    out = []
    for x in qq:
        rc = sm.D_M + 1.0 / (x * sm.V_NAV / sm.B + 1.0 / sm.A)
        t = sm.D_IN / rc
        k = math.floor(4 * t * (t + 1))
        out.append((1.5 * k * x + 1.5 + 3 * sm.V_NAV / (sm.K_NAV * rc)) ** 2)
    return np.array(out)


def ring_size(name, qq):
    out = []
    for x in qq:
        sm.use(name)
        c0 = sm.INSTANCES[name]["C"]
        sm.INSTANCES[name]["C"] = x * sm.V_NAV
        sm.use(name)
        k = er.constants()
        out.append(len(er.build(k)) if k["n_hi"] >= k["n_lo"] else np.nan)
        sm.INSTANCES[name]["C"] = c0
    return np.array(out, dtype=float)


fig, ax = plt.subplots(figsize=(8, 6))
qq = np.linspace(0.9, 4.4, 200)
for name, color, label in [("reference", plot_style.BLUE, "Reference"),
                           ("implementation", plot_style.ORANGE, "Implementation")]:
    res = np.load(f"crush_{name}_spread.npy")
    q, n = res[:, 0], res[:, 1]
    ok = ~np.isnan(n)
    slope = np.polyfit(np.log(q[ok]), np.log(n[ok]), 1)[0]
    print(f"{name}: N*={n.tolist()} slope={slope:.2f}")
    ax.loglog(qq, theorem_bound(name, qq), color=color, lw=2, label=f"{label}: bound")
    ax.loglog(qq, ring_size(name, qq), "--", color=color, lw=2, label=f"{label}: rings")
    ax.loglog(q[ok], n[ok], "o", color=color, ms=9, label=f"{label}: simulation")
    # N* above the largest tested N
    ax.loglog(q[~ok], np.full((~ok).sum(), 120.0), "^", color=color, ms=10, mfc="white", mew=2)
ax.set_xticks([1, 1.5, 2, 3, 4])
ax.set_xticklabels(["1", "1.5", "2", "3", "4"])
ax.minorticks_off()
ax.set_xlabel(r"$C/v_{nav}$")
ax.set_ylabel(r"$N^*$")
ax.legend(fontsize=10, handlelength=1.5, borderpad=0.3, labelspacing=0.25, handletextpad=0.4)
plot_style.bold_ticks(ax)
fig.tight_layout()
fig.savefig("fig_crush.png")
fig.savefig("fig_crush.pdf")
