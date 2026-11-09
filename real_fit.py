"""
real_fit.py -- Bayesian calibration of the compartment model on an injury-score time course
(GSE182939 in inputs/, GSE269622 in inputs_B/ via INPUTS=inputs_B TPS=4h,12h,48h).

Usage:  python real_fit.py <variant> [holdout_timepoint]
  variant : full     per-macro-type beta, rho (prior centred on Table 1), D_D, D_D vascular
            table1   beta, rho, D_D FIXED at Table 1 (literature parameters of the CIBB 2026 short paper); only tau, s, noise are inferred
            uniform  one beta, one rho, one D_D for all compartments (no cell-type hierarchy)
  holdout : e.g. 6wk -> that timepoint is excluded from the likelihood and predicted afterwards

Data handling (decided before looking at fit results):
  * graph = W_c from the sham section (15 nodes)
  * observed compartments: n_spots >= 10 both at the timepoint and in sham
  * y_ct = (mean_score_ct - mean_score_c,sham), then centred on the spot-weighted mean over the observed
    compartments of that timepoint (same centring is applied to the model: z_ct = pD_c - weighted mean pD)
  * noise: sd = sqrt(kappa^2 * se_ct^2 + sigma0^2), se from spot-level sd (injury + sham), kappa and sigma0 free
  * time: hours = tau * model_time (tau inferred), seeds = outer-medulla compartments at p_D0 = 0.05
"""
import os, sys, json, time
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")
import numpy as np, pandas as pd, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax_model as jm

MACROS = ["PT", "DCT", "TAL", "vascular", "immune", "other"]
T1 = {"PT": (0.40, 0.08), "DCT": (0.28, 0.10), "TAL": (0.25, 0.12), "vascular": (0.30, 0.10),
      "immune": (0.20, 0.25), "other": (0.22, 0.10)}
INP = os.environ.get("INPUTS", "inputs")          # folder with injury_long.csv, W_c_sham.csv, markers_long.csv
TPS = os.environ.get("TPS", "4h,12h,48h,6wk").split(",")
HOURS = np.array([{"4h": 4.0, "12h": 12.0, "48h": 48.0, "6wk": 1008.0}[t] for t in TPS]); NMIN = 10
DSET = "" if INP == "inputs" else "_" + INP.replace("inputs_", "")
SCORE = os.environ.get("SCORE", "inj")   # inj | noVC (Havcr1+Lcn2 only, from markers_long)
NW = int(os.environ.get("NW", 400)); NS = int(os.environ.get("NS", 400)); SEED = int(os.environ.get("SEED", 1))
SYN = os.environ.get("SYN", "none")      # none | null (uniform truth) | table1 (hierarchy truth)
PSD = float(os.environ.get("PRIOR_SD", 0.5))   # prior s.d. of log beta / log rho (full variant); 0.5 = default
PERM = int(os.environ.get("PERM", 0))          # >0: random relabelling of the graph nodes (seed = PERM); 0 = real graph

inj = pd.read_csv(f"{INP}/injury_long.csv")
Wdf = pd.read_csv(f"{INP}/W_c_sham.csv", index_col=0)
NAMES = list(Wdf.index); N = len(NAMES)
macro_idx = np.array([MACROS.index(x.split("_")[0]) for x in NAMES])
is_vasc = np.array([x.startswith("vascular") for x in NAMES])
ZONES = ["cortex", "outer_medulla", "inner_medulla"]
zone_idx = np.array([ZONES.index(x.split("_", 1)[1]) for x in NAMES])
pD0 = jnp.array([0.05 if "outer_medulla" in x else 0.0 for x in NAMES])
DH = jnp.full(N, 0.05)
Wmat = Wdf.values.copy()
if PERM > 0:   # same graph, nodes assigned to the wrong compartments: keeps weights/degrees, breaks spatial correspondence
    pi = np.random.default_rng(PERM).permutation(N); Wmat = Wmat[np.ix_(pi, pi)]
Wt, L = jm.graph_operators(Wmat)
if SCORE == "noVC":
    mk = pd.read_csv(f"{INP}/markers_long.csv")
    alt = mk[mk.gene.isin(["Havcr1", "Lcn2"])].groupby(["timepoint", "compartment"]).mean_expr.mean().rename("alt").reset_index()
    inj = inj.merge(alt, on=["timepoint", "compartment"], how="left")
    inj["mean_score"] = inj["alt"]            # sd_score kept from the original score (used only as a noise-scale proxy; kappa/sigma0 are free)
sham = inj[inj.timepoint == "sham"].set_index("compartment")

def build_data():
    Y = np.zeros((len(TPS), N)); SE = np.ones((len(TPS), N)); M = np.zeros((len(TPS), N), bool); Wn = np.zeros((len(TPS), N))
    for i, tp in enumerate(TPS):
        d = inj[inj.timepoint == tp].set_index("compartment")
        for j, c in enumerate(NAMES):
            if c in d.index and c in sham.index and d.loc[c, "n_spots"] >= NMIN and sham.loc[c, "n_spots"] >= NMIN:
                M[i, j] = True
                Y[i, j] = d.loc[c, "mean_score"] - sham.loc[c, "mean_score"]
                SE[i, j] = np.sqrt(d.loc[c, "sd_score"] ** 2 / d.loc[c, "n_spots"] + sham.loc[c, "sd_score"] ** 2 / sham.loc[c, "n_spots"])
                Wn[i, j] = d.loc[c, "n_spots"]
        w = Wn[i] / Wn[i].sum(); Wn[i] = w
        Y[i, M[i]] -= (w * Y[i]).sum()                       # centre on weighted mean of observed compartments
    return Y, SE, M, Wn
Y, SE, M, WN = build_data()

def zmodel(beta, rho, DD, tau, Wt_=None, L_=None, p0=None):
    idx = jnp.arange(len(HOURS))
    ts = jnp.minimum(jnp.asarray(HOURS) / tau, 400.0 + 0.01 * idx)
    _, pD, _ = jm.simulate_ode(Wt if Wt_ is None else Wt_, L if L_ is None else L_, beta, rho, DH, DD, pD0 if p0 is None else p0, ts)
    return pD - (jnp.asarray(WN) * pD).sum(axis=1, keepdims=True)   # same weighted centring as the data

def make_model(variant, obs_mask):
    t1b = jnp.array([T1[m][0] for m in MACROS]); t1r = jnp.array([T1[m][1] for m in MACROS])
    def model(y=None):
        if variant == "full":
            lb = numpyro.sample("logbeta", dist.Normal(jnp.log(t1b), PSD))
            lr = numpyro.sample("logrho", dist.Normal(jnp.log(t1r), PSD))
            lD = numpyro.sample("logDD", dist.Normal(jnp.log(0.2), 0.5))
            lDv = numpyro.sample("logDDv", dist.Normal(jnp.log(0.35), 0.5))
            beta, rho = jnp.exp(lb)[macro_idx], jnp.exp(lr)[macro_idx]
            DD = jnp.where(is_vasc, jnp.exp(lDv), jnp.exp(lD))
        elif variant == "zone":       # reduced model: beta, rho per ANATOMICAL ZONE, shared across cell types
            lb = numpyro.sample("logbeta", dist.Normal(jnp.log(0.30), 0.5).expand([3]))
            lr = numpyro.sample("logrho", dist.Normal(jnp.log(0.10), 0.5).expand([3]))
            lD = numpyro.sample("logDD", dist.Normal(jnp.log(0.2), 0.5))
            beta, rho, DD = jnp.exp(lb)[zone_idx], jnp.exp(lr)[zone_idx], jnp.full(N, jnp.exp(lD))
        elif variant == "nograph":   # null: no propagation. W~ = identity (self-exposure only), no diffusion, all compartments seeded alike
            lb = numpyro.sample("logbeta", dist.Normal(jnp.log(0.30), 0.5).expand([3]))
            lr = numpyro.sample("logrho", dist.Normal(jnp.log(0.10), 0.5).expand([3]))
            beta, rho, DD = jnp.exp(lb)[zone_idx], jnp.exp(lr)[zone_idx], jnp.zeros(N)
        elif variant == "uniform":
            lb = numpyro.sample("logbeta", dist.Normal(jnp.log(0.30), 0.5))
            lr = numpyro.sample("logrho", dist.Normal(jnp.log(0.10), 0.5))
            lD = numpyro.sample("logDD", dist.Normal(jnp.log(0.2), 0.5))
            beta, rho, DD = jnp.full(N, jnp.exp(lb)), jnp.full(N, jnp.exp(lr)), jnp.full(N, jnp.exp(lD))
        else:  # table1
            beta, rho = t1b[macro_idx], t1r[macro_idx]
            DD = jnp.where(is_vasc, 0.35, 0.20)
        lt = numpyro.sample("logtau", dist.Normal(0.0, 2.0))
        ls = numpyro.sample("logs", dist.Normal(jnp.log(5.0), 1.5))
        lk = numpyro.sample("logkappa", dist.Normal(0.0, 1.0))
        s0 = numpyro.sample("sigma0", dist.HalfNormal(0.1))
        if variant == "nograph":
            z = zmodel(beta, rho, DD, jnp.exp(lt), Wt_=jnp.eye(N), L_=jnp.zeros((N, N)), p0=jnp.full(N, 0.05))
        else:
            z = zmodel(beta, rho, DD, jnp.exp(lt))
        numpyro.deterministic("z", z)
        sd = jnp.sqrt((jnp.exp(lk) * jnp.asarray(SE)) ** 2 + s0 ** 2)
        with numpyro.handlers.mask(mask=jnp.asarray(obs_mask)):
            numpyro.sample("y", dist.Normal(jnp.exp(ls) * z, sd), obs=y)
    return model

if __name__ == "__main__":
    variant = sys.argv[1]; hold = sys.argv[2] if len(sys.argv) > 2 else None
    obs_mask = M.copy()
    Ydata = Y
    if SYN != "none":
        rng = np.random.default_rng(7)
        if SYN == "null":
            beta_t, rho_t, DD_t = jnp.full(N, 0.30), jnp.full(N, 0.10), jnp.full(N, 0.20)
        else:
            beta_t = jnp.array([T1[MACROS[i]][0] for i in macro_idx]); rho_t = jnp.array([T1[MACROS[i]][1] for i in macro_idx])
            DD_t = jnp.where(is_vasc, 0.35, 0.20)
        z_t = np.array(zmodel(beta_t, rho_t, DD_t, 25.0))
        Ydata = 5.0 * z_t + rng.normal(0, np.sqrt(SE ** 2 + 0.07 ** 2))
        Ydata = np.where(M, Ydata, 0.0)
        print(f"synthetic truth '{SYN}': signal max {np.abs(5*z_t[M]).max():.3f}")
    if hold: obs_mask[TPS.index(hold)] = False
    tag = f"{variant}" + (f"_hold{hold}" if hold else "") + ("" if SCORE == "inj" else f"_{SCORE}") + ("" if SYN == "none" else f"_syn{SYN}") + ("" if PSD == 0.5 else f"_psd{PSD:g}") + ("" if PERM == 0 else f"_perm{PERM}") + DSET + ("" if (SEED == 1 and NS == 400) else f"_s{SEED}n{NS}")
    model = make_model(variant, obs_mask)
    mcmc = MCMC(NUTS(model, target_accept_prob=0.95), num_warmup=NW, num_samples=NS, num_chains=2, chain_method="parallel", progress_bar=False)
    t0 = time.time(); mcmc.run(jax.random.PRNGKey(SEED), y=jnp.array(Ydata)); jax.block_until_ready(mcmc.get_samples())
    S = mcmc.get_samples(group_by_chain=False); Sg = mcmc.get_samples(group_by_chain=True)
    div = int(np.array(mcmc.get_extra_fields()["diverging"]).sum())
    from numpyro.diagnostics import summary
    sm = summary({k: v for k, v in Sg.items() if k != "z"}, group_by_chain=True)
    out = dict(tag=tag, runtime_s=round(time.time() - t0), divergences=div, n_obs=int(obs_mask.sum()), prior_sd=PSD, perm=PERM, params={})
    for k, v in sm.items():
        for i, (m_, sd_, rh, es) in enumerate(zip(np.atleast_1d(v["mean"]), np.atleast_1d(v["std"]), np.atleast_1d(v["r_hat"]), np.atleast_1d(v["n_eff"]))):
            lab = k if np.ndim(v["mean"]) == 0 else f"{k}[{(MACROS if variant=='full' else ZONES)[i] if k in ('logbeta','logrho') and variant in ('full','zone','nograph') else i}]"
            out["params"][lab] = dict(mean=float(m_), sd=float(sd_), r_hat=float(rh), ess=float(es))
    if variant == "full":
        lb, lr = np.array(S["logbeta"]), np.array(S["logrho"]); R = np.exp(lb - lr)
        out["R_diff"] = {m: [float(np.percentile(R[:, i], q)) for q in (5, 50, 95)] for i, m in enumerate(MACROS)}
        pri = np.random.default_rng(0); pb = np.exp(pri.normal(np.log([T1[m][0] for m in MACROS]), PSD, (20000, 6)) - pri.normal(np.log([T1[m][1] for m in MACROS]), PSD, (20000, 6)))
        i = {m: k for k, m in enumerate(MACROS)}
        for a, b in [("PT", "TAL"), ("PT", "DCT"), ("PT", "vascular"), ("vascular", "TAL")]:
            out[f"P(R_{a}>R_{b})"] = dict(prior=float((pb[:, i[a]] > pb[:, i[b]]).mean()), posterior=float((R[:, i[a]] > R[:, i[b]]).mean()))
    # posterior predictive / hold-out
    z = np.array(S["z"]); ls = np.array(S["logs"]); pred = np.exp(ls)[:, None, None] * z
    rows = []
    for i, tp in enumerate(TPS):
        m = M[i]
        if not m.any(): continue
        p = pred[:, i, :][:, m].mean(axis=0); y = Ydata[i, m]
        r = float(np.corrcoef(p, y)[0, 1]) if m.sum() > 2 else float("nan")
        rows.append(dict(tp=tp, held_out=bool(hold == tp), n=int(m.sum()), rmse=float(np.sqrt(np.mean((p - y) ** 2))), rmse_null_zero=float(np.sqrt(np.mean(y ** 2))), corr=r))
    out["fit_by_timepoint"] = rows
    json.dump(out, open(f"result_{tag}.json", "w"), indent=1)
    np.savez_compressed(f"samples_{tag}.npz", **{k: np.array(v) for k, v in S.items()})
    print("DONE", tag, "runtime", out["runtime_s"], "div", div)
