# Calibrating and stress-testing a graph diffusion model on spatial transcriptomics time courses

Code and results for the chapter *"Calibrating and Stress-Testing a Graph Diffusion Model on Spatial Transcriptomics Time Courses:
A Bayesian Workflow Applied to Kidney Injury"*
(C. F. Perri, A. Defilippo, F. M. Giorgi, P. H. Guzzi, P. Veltri; in preparation, Springer LNBI),
an extension of the CIBB 2026 short paper *"A Multilayer Graph Diffusion Model for Spatial Transcriptomics
Reveals Cell-Cell Propagation Dynamics"*.

The compartment-level H/D/R diffusion-reaction model is re-implemented as a differentiable simulator (JAX + Diffrax)
and calibrated with NUTS (NumPyro) on two mouse kidney ischaemia-reperfusion Visium time courses:

| cohort | GEO | time points | folder |
|---|---|---|---|
| A | GSE182939 (Dixon et al., JASN 2022) | sham, 4 h, 12 h, 48 h, 6 wk | `inputs/` |
| B | GSE269622 (Xuanyuan et al., Nat Commun 2025) | sham, 4 h, 12 h, 48 h | `inputs_B/` |

## Quick start: reproduce tables and figures from the stored results (seconds)

```bash
pip install -r requirements.txt
python make_tables.py      # held-out skill table  -> results/holdout_summary.json
python make_figs.py        # Figures 1-4           -> figures/
python test_equivalence.py # JAX model vs. original Euler implementation
```

## Repository layout

| file | content |
|---|---|
| `jax_model.py` | differentiable compartment model (ODE with continuous renormalisation; exact Euler replica) |
| `diffusion_model_reference.py` | original explicit-Euler implementation, used only by `test_equivalence.py` |
| `real_fit.py` | Bayesian calibration of one cohort: variants `full`, `zone`, `uniform`, `table1`, `nograph`; optional held-out time point |
| `joint_fit.py` | joint calibration of both cohorts (shared beta, rho, D_D; cohort-specific tau, s, kappa, sigma0) |
| `design_power.py` | synthetic design analysis (replicate sections x number of time points) |
| `make_tables.py`, `make_figs.py` | tables and figures of the chapter |
| `make_validation.py`, `cross_cohort.py` | prior sensitivity / graph permutation summary; transfer of calibrated models between cohorts |
| `preprocessing/` | from raw Visium data to the CSV files in `inputs*/` (label transfer, zoning, compartment graph, injury score) |
| `scripts/` | run scripts for cohort A, cohort B and the controls |
| `inputs/`, `inputs_B/` | per-compartment injury scores, marker means, sham compartment graph and (cohort B) QC, derived from GEO data |
| `results/` | every `result_*.json` used in the chapter (posterior summaries, diagnostics, fit per time point) |

## Re-running the inference

All commands from the repository root. Run time on a laptop: minutes per fit for cohort B, up to about one hour per fit
for the full model on cohort A.

```bash
sh scripts/run_cohort_A.sh 1        # seed 1, all variants (600+600 draws, 2 chains)
sh scripts/run_cohort_A.sh 2 "full uniform table1"
sh scripts/run_cohort_B.sh          # all variants, seeds 1-3 (1000+1000 draws)
sh scripts/run_controls.sh          # synthetic controls, score without Vcam1/Ccl2, design analysis, joint fit
sh scripts/run_validation.sh B 20   # prior sensitivity + 20 graph permutations (run from the root; outputs in the root)
python make_validation.py results   # summary of the validation runs
```

Single fits: `python real_fit.py <variant> [held-out time point]`, with environment variables
`INPUTS` (default `inputs`), `TPS` (default `4h,12h,48h,6wk`), `NW`/`NS` (warm-up/draws), `SEED`,
`SYN=null|table1` (synthetic data on the real design), `SCORE=noVC` (score without *Vcam1* and *Ccl2*), `PRIOR_SD` (prior s.d. of log beta / log rho, default 0.5),
`PERM=k` (graph with nodes randomly relabelled, seed k).
Output file names encode variant, held-out time point, cohort (`_B`), seed and number of draws, e.g.
`result_full_hold12h_B_s2n1000.json`.

## From raw data (optional)

1. Download the Space Ranger outputs of GSE182939 and GSE269622 and the single-cell matrix of GSE107585 from GEO.
2. `python preprocessing/build_reference.py Mouse_kidney_single_cell_datamatrix.txt data/GSE107585_sc.h5ad`
3. Edit `SAMPLE_PATHS` in `preprocessing/preprocess_compartments.py` (cohort A) and `DATA_DIR` in
   `preprocessing/export_for_inference_B.py` (cohort B) to your local layout.
4. `python preprocessing/export_for_inference.py` and `python preprocessing/export_for_inference_B.py`;
   copy the resulting CSV files into `inputs/` and `inputs_B/`.

## Notes

* Priors, the observation model and the evaluation (skill = 1 - RMSE/RMSE of the zero predictor on the held-out
  time point) are described in the chapter; the code is the reference.
* Sampling diagnostics (divergences, R-hat, ESS) are stored in every result file. Fits of the `zone` and `nograph`
  variants and the time scale `logtau` on cohort A mix poorly; the chapter flags these results.
