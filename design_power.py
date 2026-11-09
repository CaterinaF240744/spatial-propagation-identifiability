"""
design_power.py -- experimental-design (power) analysis on SYNTHETIC data built on the real kidney graph.
Question: how much data would be needed to tell a cell-type hierarchy (Table 1) from no hierarchy?

    python design_power.py <truth: table1|null> <f> <ntp: 4|8>

truth : table1 -> data generated with the Table 1 beta/rho per cell type;  null -> beta=0.30, rho=0.10 for all
f     : noise multiplier. f = 1/sqrt(R) mimics R replicate sections per timepoint (R=1: f=1; R=4: 0.5; R=16: 0.25).
        Both noise components (spot-level SE and sigma0 = section-level) are divided by sqrt(R) -- an assumption.
ntp   : 4 -> hours 4,12,48,1008 (the real design);  8 -> 2,4,8,12,24,48,168,1008
Fit: the 'full' model of real_fit.py (beta, rho per cell type, priors centred on Table 1), same observation model.
"""
import os, sys, json, time
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import real_fit as R          # graph, seeds, compartment indices, spot counts, T1, MACROS (real data only used for these)
import jax_model as jm

truth, f, ntp = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
HRS = np.array({4: [4, 12, 48, 1008], 8: [2, 4, 8, 12, 24, 48, 168, 1008]}[ntp], float)
obs = R.M[1].copy()                                      # compartments observed at 12h in the real data (12 of 15)
w = R.WN[1].copy(); w = w / w.sum()                      # spot-count weights for the centring
se = np.where(obs, R.SE[1], 1.0)
TAU, S, SIG0 = 25.0, 5.0, 0.07                           # same synthetic truth as the earlier control (syn_null / syn_table1)

def zmodel(beta, rho, DD, tau):
    ts = jnp.minimum(jnp.asarray(HRS) / tau, 400.0 + 0.01 * jnp.arange(len(HRS)))
    _, pD, _ = jm.simulate_ode(R.Wt, R.L, beta, rho, R.DH, DD, R.pD0, ts)
    return pD - (jnp.asarray(w) * pD).sum(axis=1, keepdims=True)

if truth == "null":
    b_t, r_t, D_t = jnp.full(R.N, 0.30), jnp.full(R.N, 0.10), jnp.full(R.N, 0.20)
else:
    b_t = jnp.array([R.T1[R.MACROS[i]][0] for i in R.macro_idx]); r_t = jnp.array([R.T1[R.MACROS[i]][1] for i in R.macro_idx])
    D_t = jnp.where(R.is_vasc, 0.35, 0.20)
rng = np.random.default_rng(11)
z_t = np.array(zmodel(b_t, r_t, D_t, TAU))
Y = S * z_t + rng.normal(0, 1, z_t.shape) * f * np.sqrt(se[None, :] ** 2 + SIG0 ** 2)
mask = np.tile(obs, (len(HRS), 1)); Y = np.where(mask, Y, 0.0)

t1b = jnp.array([R.T1[m][0] for m in R.MACROS]); t1r = jnp.array([R.T1[m][1] for m in R.MACROS])
def model(y=None):
    lb = numpyro.sample("logbeta", dist.Normal(jnp.log(t1b), 0.5)); lr = numpyro.sample("logrho", dist.Normal(jnp.log(t1r), 0.5))
    lD = numpyro.sample("logDD", dist.Normal(jnp.log(0.2), 0.5)); lDv = numpyro.sample("logDDv", dist.Normal(jnp.log(0.35), 0.5))
    beta, rho = jnp.exp(lb)[R.macro_idx], jnp.exp(lr)[R.macro_idx]; DD = jnp.where(R.is_vasc, jnp.exp(lDv), jnp.exp(lD))
    lt = numpyro.sample("logtau", dist.Normal(0.0, 2.0)); ls = numpyro.sample("logs", dist.Normal(jnp.log(5.0), 1.5))
    lk = numpyro.sample("logkappa", dist.Normal(0.0, 1.0)); s0 = numpyro.sample("sigma0", dist.HalfNormal(0.1))
    z = zmodel(beta, rho, DD, jnp.exp(lt))
    sd = jnp.sqrt((jnp.exp(lk) * jnp.asarray(se)) ** 2 + s0 ** 2)
    with numpyro.handlers.mask(mask=jnp.asarray(mask)):
        numpyro.sample("y", dist.Normal(jnp.exp(ls) * z, sd), obs=y)

if __name__ == "__main__":
    tag = f"design_{truth}_f{f}_k{ntp}"
    mcmc = MCMC(NUTS(model, target_accept_prob=0.95), num_warmup=400, num_samples=400, num_chains=2, chain_method="parallel", progress_bar=False)
    t0 = time.time(); mcmc.run(jax.random.PRNGKey(2), y=jnp.array(Y)); S_ = mcmc.get_samples(); jax.block_until_ready(S_)
    div = int(np.array(mcmc.get_extra_fields()["diverging"]).sum())
    lb, lr = np.array(S_["logbeta"]), np.array(S_["logrho"]); Rd = np.exp(lb - lr); I = {m: k for k, m in enumerate(R.MACROS)}
    out = dict(tag=tag, truth=truth, f=f, ntp=ntp, runtime_s=round(time.time() - t0), divergences=div,
               sd_over_prior_logbeta=dict(zip(R.MACROS, (lb.std(axis=0) / 0.5).round(3).tolist())),
               sd_over_prior_logrho=dict(zip(R.MACROS, (lr.std(axis=0) / 0.5).round(3).tolist())),
               sd_over_prior_logR=dict(zip(R.MACROS, ((lb - lr).std(axis=0) / np.sqrt(0.5 ** 2 + 0.5 ** 2)).round(3).tolist())),
               R_diff_median=dict(zip(R.MACROS, np.median(Rd, axis=0).round(2).tolist())),
               R_diff_true=dict(zip(R.MACROS, np.exp(np.log([R.T1[m][0] for m in R.MACROS]) - np.log([R.T1[m][1] for m in R.MACROS])).round(2).tolist())) if truth == "table1" else "all equal (3.0)",
               logtau_mean=float(np.array(S_["logtau"]).mean()), logtau_true=float(np.log(TAU)))
    for a, b in [("PT", "TAL"), ("PT", "DCT"), ("vascular", "TAL")]:
        out[f"P(R_{a}>R_{b})"] = float((Rd[:, I[a]] > Rd[:, I[b]]).mean())
    json.dump(out, open(f"result_{tag}.json", "w"), indent=1); print("DONE", tag, out["runtime_s"], "s div", div)
