"""
jax_model.py -- differentiable version of the compartment-level (H, D, R) model
of diffusion_model_reference.py (the original explicit-Euler implementation).

Same equations (mean-field, mu = 0):
    lam   = W_tilde @ pD
    dpH   = -beta * lam * pH - D_H * (L @ pH)
    dpD   =  beta * lam * pH - rho * pD - D_D * (L @ pD)
    dpR   =  rho * pD

Two integrators:
  * simulate_euler : bit-for-bit replica of CompartmentDiffusionModel.simulate
                     (explicit Euler, clip + renormalise). Used ONLY to check
                     that the JAX port reproduces the published code.
  * simulate_ode   : diffrax Tsit5 without clipping, smooth in the parameters.
                     Used for inference (gradients for NUTS).
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax


def graph_operators(W):
    """Row-normalised W_tilde and Laplacian L, as in CompartmentDiffusionModel."""
    W = jnp.asarray(W, dtype=jnp.float64)
    row = W.sum(axis=1, keepdims=True)
    row = jnp.where(row == 0, 1.0, row)
    return W / row, jnp.diag(row.flatten()) - W


def _rhs(t, y, args):
    """ODE right-hand side. All parameters arrive through `args` (not closed over),
    which diffrax requires for correct gradients under vmap/pmap."""
    Wt, L, beta, rho, D_H, D_D, renorm = args
    n = beta.shape[0]
    pH, pD = y[:n], y[n:2 * n]
    lam = Wt @ pD
    inf = beta * lam * pH
    dH = -inf - D_H * (L @ pH)
    dD = inf - rho * pD - D_D * (L @ pD)
    dR = rho * pD
    d = jnp.concatenate([dH, dD, dR])
    # Continuous-time limit of the per-step renormalisation in
    # CompartmentDiffusionModel.simulate (p <- p / sum(p)): with sum(p)=1 one
    # step gives p' = (p + dt f)/(1 + dt*sum f), hence dp/dt = f - p * sum_k f_k.
    ds = dH + dD + dR
    return d - renorm * y * jnp.tile(ds, 3)


def simulate_ode(Wt, L, beta, rho, D_H, D_D, p_D0, ts, rtol=1e-6, atol=1e-9, renorm=True):
    """Return (pH, pD, pR) at times ts (model units), each of shape (len(ts), n)."""
    n = beta.shape[0]
    y0 = jnp.concatenate([1.0 - p_D0, p_D0, jnp.zeros(n)])
    args = (Wt, L, beta, rho, D_H, D_D, jnp.float64(1.0 if renorm else 0.0))
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(_rhs), diffrax.Tsit5(), t0=0.0, t1=ts[-1], dt0=0.05, y0=y0, args=args,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        max_steps=2048, adjoint=diffrax.DirectAdjoint(), throw=False,
    )
    # throw=False: a failed solve (extreme parameters proposed by NUTS) becomes NaN,
    # i.e. the proposal is rejected as a divergence instead of aborting the run.
    ys = jnp.where(sol.result == diffrax.RESULTS.successful, sol.ys, jnp.nan)
    return ys[:, :n], ys[:, n:2 * n], ys[:, 2 * n:]


def simulate_euler(W, beta, rho, D_H, D_D, p_D0, t_end=60.0, dt=0.05):
    """Replica of CompartmentDiffusionModel.simulate (Euler + clip + renorm)."""
    Wt, L = graph_operators(W)
    n = beta.shape[0]
    steps = int(jnp.ceil(t_end / dt)) + 1

    def step(carry, _):
        pH, pD, pR = carry
        lam = Wt @ pD
        dH = -beta * lam * pH - D_H * (L @ pH)
        dD = beta * lam * pH - rho * pD - D_D * (L @ pD)
        dR = rho * pD
        pH2 = jnp.clip(pH + dt * dH, 0.0, 1.0)
        pD2 = jnp.clip(pD + dt * dD, 0.0, 1.0)
        pR2 = jnp.clip(pR + dt * dR, 0.0, 1.0)
        s = pH2 + pD2 + pR2
        s = jnp.where(s == 0, 1.0, s)
        new = (pH2 / s, pD2 / s, pR2 / s)
        return new, new

    init = (1.0 - p_D0, p_D0, jnp.zeros(n))
    _, (H, D, R) = jax.lax.scan(step, init, None, length=steps - 1)
    H = jnp.vstack([init[0], H]); D = jnp.vstack([init[1], D]); R = jnp.vstack([init[2], R])
    time = jnp.linspace(0.0, t_end, steps)
    return time, H, D, R
