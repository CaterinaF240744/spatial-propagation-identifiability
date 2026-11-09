"""cross_cohort.py -- transfer test: parameters (incl. time scale) learned on one cohort predict the spatial pattern of the other.
Metric: Pearson correlation across observed compartments between predicted centred pD and observed centred score
(scale-free, so the cohort-specific score scale s is not needed). Baseline: the other cohort's OBSERVED pattern at the
same time point (same compartment names), i.e. 'the tissues respond alike' without any model."""
import os, sys, json, numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import jax_model as jm, joint_fit as jf
MACROS = jf.MACROS; T1 = jf.T1; D = jf.D
rng = np.random.default_rng(0)
FILES = {("A", "uniform"): "samples_uniform.npz", ("A", "table1"): "samples_table1.npz", ("A", "full"): "samples_full.npz",
         ("B", "uniform"): "samples_uniform_B.npz", ("B", "table1"): "samples_table1_B.npz", ("B", "full"): "samples_full_B.npz"}
def predict(src, var, tgt, ndraw=200):
    S = np.load(FILES[(src, var)]); d = D[tgt]; n = len(S["logtau"]); idx = rng.choice(n, ndraw, replace=False)
    out = []
    sim = jax.jit(lambda b, r, dd, ts: jm.simulate_ode(d["Wt"], d["L"], b, r, jnp.full(d["N"], 0.05), dd, d["p0"], ts)[1])
    for k in idx:
        if var == "uniform":
            b = jnp.full(d["N"], np.exp(S["logbeta"][k])); r = jnp.full(d["N"], np.exp(S["logrho"][k])); dd = jnp.full(d["N"], np.exp(S["logDD"][k]))
        elif var == "table1":
            b = jnp.array([T1[MACROS[i]][0] for i in d["mi"]]); r = jnp.array([T1[MACROS[i]][1] for i in d["mi"]])
            dd = jnp.where(d["iv"], 0.35, 0.2)   # Table 1 diffusion values used by real_fit table1
        else:
            b = jnp.exp(S["logbeta"][k])[d["mi"]]; r = jnp.exp(S["logrho"][k])[d["mi"]]
            dd = jnp.where(d["iv"], np.exp(S["logDDv"][k]), np.exp(S["logDD"][k]))
        ts = jnp.asarray(d["hours"]) / np.exp(S["logtau"][k])
        pD = np.array(sim(b, r, dd, ts)); out.append(pD - (d["WN"] * pD).sum(1, keepdims=True))
    return np.nanmean(out, axis=0)
res = {}
for src, tgt in (("A", "B"), ("B", "A")):
    dt = D[tgt]; ds = D[src]
    for var in ("uniform", "table1", "full"):
        z = predict(src, var, tgt)
        for i, tp in enumerate(dt["tps"]):
            m = dt["M"][i]
            res[f"{src}->{tgt}|{var}|{tp}"] = float(np.corrcoef(z[i, m], dt["Y"][i, m])[0, 1])
    for i, tp in enumerate(dt["tps"]):        # baseline: other cohort's observed pattern, same compartments
        if tp not in ds["tps"]: continue
        j = ds["tps"].index(tp); ys = dict(zip(ds["names"], np.where(ds["M"][j], ds["Y"][j], np.nan)))
        m = dt["M"][i] & np.array([np.isfinite(ys.get(n, np.nan)) for n in dt["names"]])
        res[f"{src}->{tgt}|observed|{tp}"] = float(np.corrcoef([ys[n] for n, k in zip(dt["names"], m) if k], dt["Y"][i, m])[0, 1]); res[f"{src}->{tgt}|n_shared|{tp}"] = int(m.sum())
json.dump(res, open("results/cross_cohort.json", "w"), indent=1)
for k, v in res.items(): print(f"{k:35s} {v:+.2f}" if isinstance(v, float) else f"{k:35s} {v}")
