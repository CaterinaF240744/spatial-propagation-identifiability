"""
joint_fit.py -- joint Bayesian calibration on BOTH cohorts (A = GSE182939, B = GSE269622).
Shared across cohorts: beta, rho (per cell type, or uniform), D_D.   Cohort-specific: tau, s, kappa, sigma0, graph (each from its own sham).
Usage: python joint_fit.py <full|uniform>      env: NW, NS, SEED
Output: result_joint_<variant>_s<SEED>n<NS>.json  (identification summary: sd/prior, P(R_a>R_b), tau per cohort, fit per cohort/timepoint)
"""
import os, sys, json, time
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")
import numpy as np, pandas as pd, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax_model as jm

MACROS = ["PT", "DCT", "TAL", "vascular", "immune", "other"]
T1 = {"PT": (0.40, 0.08), "DCT": (0.28, 0.10), "TAL": (0.25, 0.12), "vascular": (0.30, 0.10), "immune": (0.20, 0.25), "other": (0.22, 0.10)}
HRS = {"4h": 4.0, "12h": 12.0, "48h": 48.0, "6wk": 1008.0}
NMIN = 10
COH = {"A": ("inputs", ["4h", "12h", "48h", "6wk"]), "B": ("inputs_B", ["4h", "12h", "48h"])}
NW = int(os.environ.get("NW", 400)); NS = int(os.environ.get("NS", 400)); SEED = int(os.environ.get("SEED", 1))

def load(inp, tps):
    inj = pd.read_csv(f"{inp}/injury_long.csv"); Wdf = pd.read_csv(f"{inp}/W_c_sham.csv", index_col=0)
    names = list(Wdf.index); N = len(names); sham = inj[inj.timepoint == "sham"].set_index("compartment")
    Y = np.zeros((len(tps), N)); SE = np.ones((len(tps), N)); M = np.zeros((len(tps), N), bool); Wn = np.zeros((len(tps), N))
    for i, tp in enumerate(tps):
        d = inj[inj.timepoint == tp].set_index("compartment")
        for j, c in enumerate(names):
            if c in d.index and c in sham.index and d.loc[c, "n_spots"] >= NMIN and sham.loc[c, "n_spots"] >= NMIN:
                M[i, j] = True; Y[i, j] = d.loc[c, "mean_score"] - sham.loc[c, "mean_score"]
                SE[i, j] = np.sqrt(d.loc[c, "sd_score"] ** 2 / d.loc[c, "n_spots"] + sham.loc[c, "sd_score"] ** 2 / sham.loc[c, "n_spots"]); Wn[i, j] = d.loc[c, "n_spots"]
        w = Wn[i] / Wn[i].sum(); Wn[i] = w; Y[i, M[i]] -= (w * Y[i]).sum()
    Wt, L = jm.graph_operators(Wdf.values)
    mi = np.array([MACROS.index(x.split("_")[0]) for x in names]); iv = np.array([x.startswith("vascular") for x in names]); p0 = jnp.array([0.05 if "outer_medulla" in x else 0.0 for x in names])
    return dict(mi=mi, iv=iv, p0=p0, names=names, N=N, Y=Y, SE=SE, M=M, WN=Wn, Wt=Wt, L=L, hours=np.array([HRS[t] for t in tps]), tps=tps)

D = {k: load(*v) for k, v in COH.items()}

def zmodel(d, beta, rho, DD, tau):
    idx = jnp.arange(len(d["hours"])); ts = jnp.minimum(jnp.asarray(d["hours"]) / tau, 400.0 + 0.01 * idx)
    _, pD, _ = jm.simulate_ode(d["Wt"], d["L"], beta, rho, jnp.full(d["N"], 0.05), DD, d["p0"], ts)
    return pD - (jnp.asarray(d["WN"]) * pD).sum(axis=1, keepdims=True)

def make_model(variant):
    t1b = jnp.array([T1[m][0] for m in MACROS]); t1r = jnp.array([T1[m][1] for m in MACROS])
    def model(yA=None, yB=None):
        if variant == "full":
            lb = numpyro.sample("logbeta", dist.Normal(jnp.log(t1b), 0.5)); lr = numpyro.sample("logrho", dist.Normal(jnp.log(t1r), 0.5))
            lD = numpyro.sample("logDD", dist.Normal(jnp.log(0.2), 0.5)); lDv = numpyro.sample("logDDv", dist.Normal(jnp.log(0.35), 0.5))
        else:
            lb = numpyro.sample("logbeta", dist.Normal(jnp.log(0.30), 0.5)); lr = numpyro.sample("logrho", dist.Normal(jnp.log(0.10), 0.5)); lD = numpyro.sample("logDD", dist.Normal(jnp.log(0.2), 0.5))
        for c, y in (("A", yA), ("B", yB)):
            d = D[c]; n_ = d["N"]
            if variant == "full": beta, rho, DD = jnp.exp(lb)[d["mi"]], jnp.exp(lr)[d["mi"]], jnp.where(d["iv"], jnp.exp(lDv), jnp.exp(lD))
            else: beta, rho, DD = jnp.full(n_, jnp.exp(lb)), jnp.full(n_, jnp.exp(lr)), jnp.full(n_, jnp.exp(lD))
            lt = numpyro.sample(f"logtau_{c}", dist.Normal(0.0, 2.0)); ls = numpyro.sample(f"logs_{c}", dist.Normal(jnp.log(5.0), 1.5))
            lk = numpyro.sample(f"logkappa_{c}", dist.Normal(0.0, 1.0)); s0 = numpyro.sample(f"sigma0_{c}", dist.HalfNormal(0.1))
            z = zmodel(d, beta, rho, DD, jnp.exp(lt)); numpyro.deterministic(f"z_{c}", z)
            sd = jnp.sqrt((jnp.exp(lk) * jnp.asarray(d["SE"])) ** 2 + s0 ** 2)
            with numpyro.handlers.mask(mask=jnp.asarray(d["M"])):
                numpyro.sample(f"y_{c}", dist.Normal(jnp.exp(ls) * z, sd), obs=y)
    return model

if __name__ == "__main__":
    variant = sys.argv[1]; model = make_model(variant)
    mcmc = MCMC(NUTS(model, target_accept_prob=0.95), num_warmup=NW, num_samples=NS, num_chains=2, chain_method="parallel", progress_bar=False)
    t0 = time.time(); mcmc.run(jax.random.PRNGKey(SEED), yA=jnp.array(D["A"]["Y"]), yB=jnp.array(D["B"]["Y"])); jax.block_until_ready(mcmc.get_samples())
    S = mcmc.get_samples(group_by_chain=False); Sg = mcmc.get_samples(group_by_chain=True)
    div = int(np.array(mcmc.get_extra_fields()["diverging"]).sum())
    from numpyro.diagnostics import summary
    sm = summary({k: v for k, v in Sg.items() if not k.startswith("z_")}, group_by_chain=True)
    tag = f"joint_{variant}_s{SEED}n{NS}"; out = dict(tag=tag, runtime_s=round(time.time() - t0), divergences=div, params={})
    for k, v in sm.items():
        for i, (m_, sd_, rh, es) in enumerate(zip(np.atleast_1d(v["mean"]), np.atleast_1d(v["std"]), np.atleast_1d(v["r_hat"]), np.atleast_1d(v["n_eff"]))):
            lab = k if np.ndim(v["mean"]) == 0 else f"{k}[{MACROS[i] if (variant=='full' and k in ('logbeta','logrho')) else i}]"
            out["params"][lab] = dict(mean=float(m_), sd=float(sd_), r_hat=float(rh), ess=float(es))
    if variant == "full":
        lb, lr = np.array(S["logbeta"]), np.array(S["logrho"]); R = np.exp(lb - lr)
        out["R_diff"] = {m: [float(np.percentile(R[:, i], q)) for q in (5, 50, 95)] for i, m in enumerate(MACROS)}
        pri = np.random.default_rng(0); pb = np.exp(pri.normal(np.log([T1[m][0] for m in MACROS]), 0.5, (20000, 6)) - pri.normal(np.log([T1[m][1] for m in MACROS]), 0.5, (20000, 6)))
        i = {m: k for k, m in enumerate(MACROS)}
        for a, b in [("PT", "TAL"), ("PT", "DCT"), ("PT", "vascular"), ("vascular", "TAL")]:
            out[f"P(R_{a}>R_{b})"] = dict(prior=float((pb[:, i[a]] > pb[:, i[b]]).mean()), posterior=float((R[:, i[a]] > R[:, i[b]]).mean()))
    out["fit"] = {}
    for c in ("A", "B"):
        d = D[c]; z = np.array(S[f"z_{c}"]); pred = np.exp(np.array(S[f"logs_{c}"]))[:, None, None] * z; rows = []
        for i, tp in enumerate(d["tps"]):
            m = d["M"][i]; p = pred[:, i, :][:, m].mean(axis=0); y = d["Y"][i, m]
            rows.append(dict(tp=tp, n=int(m.sum()), rmse=float(np.sqrt(np.mean((p - y) ** 2))), rmse_null_zero=float(np.sqrt(np.mean(y ** 2))), corr=float(np.corrcoef(p, y)[0, 1])))
        out["fit"][c] = rows
    json.dump(out, open(f"result_{tag}.json", "w"), indent=1)
    print("DONE", tag, "runtime", out["runtime_s"], "div", div)
