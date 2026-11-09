import json, glob, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False, "savefig.dpi": 300})
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]; INK = "#3d3d3a"; GRID = "#d9d8d2"
R = "results/"; J = lambda f: json.load(open(R + f))
MACROS = ["PT", "DCT", "TAL", "vascular", "immune", "other"]
T1 = {"PT": (0.40, 0.08), "DCT": (0.28, 0.10), "TAL": (0.25, 0.12), "vascular": (0.30, 0.10), "immune": (0.20, 0.25), "other": (0.22, 0.10)}

# ---------- Fig 1: data ----------
import joint_fit as jf
ZON = ["cortex", "outer_medulla", "inner_medulla"]; ZL = ["cortex", "outer medulla", "inner medulla"]
fig, axs = plt.subplots(1, 2, figsize=(4.8, 1.9), sharey=True)
for ax, c, ttl in zip(axs, ("A", "B"), ("GSE182939", "GSE269622")):
    d = jf.D[c]; x = np.arange(len(d["tps"]))
    for j, n in enumerate(d["names"]):
        zi = ZON.index(n.split("_", 1)[1]); m = d["M"][:, j]
        if m.sum() == 0: continue
        ax.plot(x[m], d["Y"][m, j], "-o", color=C[zi], lw=0.8, ms=2.5, alpha=0.85)
    ax.axhline(0, color=GRID, lw=0.6, zorder=0); ax.set_xticks(x); ax.set_xticklabels(d["tps"]); ax.set_title(ttl, fontsize=8, color=INK)
axs[0].set_ylabel("centred injury score\n(vs sham)")
for zi, zl in enumerate(ZL): axs[1].plot([], [], "-o", color=C[zi], ms=2.5, lw=0.8, label=zl)
fig.legend(*axs[1].get_legend_handles_labels(), ncol=3, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig("figures/fig_data.pdf"); fig.savefig("figures/fig_data.png")

# ---------- Fig 2: prior vs posterior of R = beta/rho ----------
rng = np.random.default_rng(0)
pb = np.exp(rng.normal(np.log([T1[m][0] for m in MACROS]), 0.5, (20000, 6)) - rng.normal(np.log([T1[m][1] for m in MACROS]), 0.5, (20000, 6)))
prior = {m: np.percentile(pb[:, i], [5, 50, 95]) for i, m in enumerate(MACROS)}
srcs = [("prior", prior, "#9a9990"), ("GSE182939", J("result_full_s1n600.json")["R_diff"], C[0]),
        ("GSE269622", J("result_full_B_s1n1000.json")["R_diff"], C[1]), ("joint", J("result_joint_full_s1n500.json")["R_diff"], C[2])]
fig, ax = plt.subplots(figsize=(4.8, 2.0))
for k, (lab, dd, col) in enumerate(srcs):
    for i, m in enumerate(MACROS):
        lo, md, hi = dd[m]; xx = i + (k - 1.5) * 0.17
        ax.plot([xx, xx], [lo, hi], color=col, lw=2, solid_capstyle="round"); ax.plot(xx, md, "o", color=col, ms=3.5, mec="white", mew=0.6)
    ax.plot([], [], "-o", color=col, lw=2, ms=3.5, label=lab)
ax.set_yscale("log"); ax.set_xticks(range(6)); ax.set_xticklabels(MACROS); ax.set_ylabel(r"$R_{\mathrm{diff}}=\beta/\rho$ (5–50–95%)")
ax.grid(axis="y", color=GRID, lw=0.4); ax.legend(ncol=4, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
fig.tight_layout(); fig.savefig("figures/fig_posterior.pdf"); fig.savefig("figures/fig_posterior.png")

# ---------- Fig 3: hold-out skill ----------
S = json.load(open(R + "holdout_summary.json"))
V = ["full", "zone", "uniform", "table1", "nograph"]; VL = ["full", "zone", "uniform", "Table 1", "no propagation"]
fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.0), sharey=True, gridspec_kw=dict(width_ratios=[4, 3]))
for ax, c, tps, ttl in zip(axs, ("A", "B"), (["4h", "12h", "48h", "6wk"], ["4h", "12h", "48h"]), ("GSE182939", "GSE269622")):
    for vi, v in enumerate(V):
        for ti, t in enumerate(tps):
            e = S.get(f"{c}|{v}|{t}");
            if e is None: continue
            sk = e["skill"]; xx = ti + (vi - 2) * 0.13; y = max(sk, -1.0)
            ax.plot([xx, xx], [0, y], color=C[vi], lw=2.2, solid_capstyle="butt")
            if sk < -1.0: ax.text(xx, -1.0, f"{sk:.1f}", ha="center", va="top", fontsize=5.5, color=INK)
    ax.axhline(0, color=INK, lw=0.6); ax.set_xticks(range(len(tps))); ax.set_xticklabels(tps); ax.set_title(ttl, fontsize=8, color=INK)
    ax.set_ylim(-1.15, 0.8); ax.grid(axis="y", color=GRID, lw=0.4)
axs[0].set_ylabel("skill vs. zero predictor\n" + r"$1-\mathrm{RMSE}/\mathrm{RMSE}_0$ (held-out)")
for vi, vl in enumerate(VL): axs[1].plot([], [], color=C[vi], lw=2.2, label=vl)
fig.legend(*axs[1].get_legend_handles_labels(), ncol=5, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig("figures/fig_holdout.pdf"); fig.savefig("figures/fig_holdout.png")

# ---------- Fig 4: design ----------
fig, ax = plt.subplots(figsize=(4.8, 2.0))
rep = {1.0: 1, 0.5: 4, 0.25: 16}
for ti, (truth, lab) in enumerate((("table1", "hierarchy (Table 1)"), ("null", "no hierarchy"))):
    for k, ls in ((4, "-"), (8, "--")):
        xs, ys = [], []
        if k == 4: xs.append(1); ys.append(J(f"result_full_syn{truth}.json")["P(R_PT>R_TAL)"]["posterior"])
        for f in (0.5, 0.25):
            xs.append(rep[f]); ys.append(J(f"result_design_{truth}_f{f}_k{k}.json")["P(R_PT>R_TAL)"])
        ax.plot(xs, ys, ls, marker="o", ms=3.5, color=C[ti], lw=1.4, label=f"{lab}, {k} time points")
ax.axhline(0.805, color="#9a9990", lw=0.8, ls=":"); ax.text(17, 0.805, "prior", fontsize=6, va="center", color="#6b6a63")
ax.set_xscale("log", base=2); ax.set_xticks([1, 4, 16]); ax.set_xticklabels(["1", "4", "16"]); ax.set_xlim(0.8, 24)
ax.set_xlabel("sections per time point (equivalent)"); ax.set_ylabel(r"posterior $P(R_{PT}>R_{TAL})$"); ax.set_ylim(0.4, 1.02)
ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5)); ax.grid(axis="y", color=GRID, lw=0.4)
fig.tight_layout(); fig.savefig("figures/fig_design.pdf"); fig.savefig("figures/fig_design.png")
print("ok")
