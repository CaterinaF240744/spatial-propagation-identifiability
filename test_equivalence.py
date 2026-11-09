"""Check that the JAX port reproduces the repo's CompartmentDiffusionModel.
Uses a MOCK compartment graph (real W_c needs the user's data)."""
import numpy as np, jax.numpy as jnp
from diffusion_model_reference import CompartmentDiffusionModel
import jax_model as jm

TABLE1 = {  # literature parameters (Table 1 of the CIBB 2026 short paper)
    "PT": (0.40, 0.08, 0.20), "DCT": (0.28, 0.10, 0.20), "TAL": (0.25, 0.12, 0.20),
    "vascular": (0.30, 0.10, 0.35), "immune": (0.20, 0.25, 0.20), "other": (0.22, 0.10, 0.20)}
names = ["PT_cortex","PT_outer_medulla","PT_inner_medulla","DCT_cortex","DCT_outer_medulla",
         "TAL_cortex","TAL_outer_medulla","TAL_inner_medulla","vascular_cortex",
         "vascular_outer_medulla","other_cortex","other_outer_medulla","other_inner_medulla"]
rng = np.random.default_rng(0)
n = len(names); W = rng.uniform(0, 1, (n, n)) * (rng.uniform(size=(n, n)) < 0.5)
W = np.triu(W, 1); W = W + W.T
beta = np.array([TABLE1[x.split("_")[0]][0] for x in names])
rho  = np.array([TABLE1[x.split("_")[0]][1] for x in names])
DD   = np.array([TABLE1[x.split("_")[0]][2] for x in names]); DH = np.full(n, 0.05)
pD0  = np.array([0.05 if "outer_medulla" in x else 0.0 for x in names])

ref = CompartmentDiffusionModel(W, beta, rho, D_H=DH, D_D=DD, compartment_names=names)
tr = ref.simulate(pD0, t_end=60.0, dt=0.05)

t, H, D, R = jm.simulate_euler(W, jnp.array(beta), jnp.array(rho), jnp.array(DH), jnp.array(DD), jnp.array(pD0))
print("Euler port  max|pD diff| :", float(np.abs(np.array(D) - tr["pD"]).max()))
print("Euler port  max|pR diff| :", float(np.abs(np.array(R) - tr["pR"]).max()))

Wt, L = jm.graph_operators(W)
ts = jnp.linspace(0.0, 60.0, 1201)
args = (Wt, L, jnp.array(beta), jnp.array(rho), jnp.array(DH), jnp.array(DD), jnp.array(pD0), ts)
for rn in (False, True):
    H2, D2, R2 = jm.simulate_ode(*args, renorm=rn)
    print(f"[renorm={rn}] max|pD - repo Euler dt=0.05| = {float(np.abs(np.array(D2)-tr['pD']).max()):.4f}   "
          f"max|H+D+R-1| = {float(np.abs(np.array(H2+D2+R2)-1).max()):.2e}")
# isolate Euler discretisation error: repo Euler at a much finer step
tr_f = ref.simulate(pD0, t_end=60.0, dt=0.005)
H2, D2, R2 = jm.simulate_ode(*args, renorm=True)
idx = np.arange(0, len(tr_f["time"]), 10)[:1201]
print("renorm=True vs repo Euler dt=0.005 : max|pD diff| =", float(np.abs(np.array(D2) - tr_f["pD"][idx]).max()))
Dn = np.array(D2); tt = np.array(ts)
t_act = np.array([tt[np.where(Dn[:, i] > 0.10)[0][0]] if (Dn[:, i] > 0.10).any() else np.nan for i in range(n)])
m = ref.metrics(tr).set_index("compartment")["t_act"].values
print("t_act: ODE vs repo(dt=0.05) max abs diff =", float(np.nanmax(np.abs(t_act - m))), "model units; same ranking:",
      bool((np.argsort(np.argsort(np.nan_to_num(t_act,nan=99)))==np.argsort(np.argsort(np.nan_to_num(m,nan=99)))).all()))
from scipy.stats import spearmanr
ok = ~np.isnan(t_act) & ~np.isnan(m)
print("t_act Spearman(ODE, repo) =", round(float(spearmanr(t_act[ok], m[ok])[0]), 4), f"on {ok.sum()}/{n} compartments;",
      "ties in repo t_act:", int(len(m[ok]) - len(np.unique(m[ok]))))
