"""
export_for_inference.py -- GSE182939: run the preprocessing on every section and write the small CSV files
used by the inference code (inference_inputs/, copy them to inputs/).  Run from the repository root:

    python preprocessing/export_for_inference.py

Output: W_c_<tp>.csv (compartment graph), names_<tp>.csv, injury_long.csv (per timepoint x compartment:
n_spots, mean/sd of the scanpy injury score), markers_long.csv (mean expression per marker gene),
qc.csv (label-transfer KL, n_spots, tissue-wide mean score).  Re-running skips finished timepoints.
"""
import os, re, sys
import numpy as np
import pandas as pd


def hours_of(tp, table=None):
    """Hours post-injury from the sample label ('sham'->0, '4h'->4, '2d'/'2dps'->48, '6wk'->1008).
    The local TIMEPOINT_HOURS table is used only if it has the key; otherwise the label is parsed."""
    if table and tp in table:
        return float(table[tp])
    t = tp.lower().strip()
    if t == "sham":
        return 0.0
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([a-z]+)", t)
    if not m:
        raise ValueError(f"Cannot parse hours from timepoint label '{tp}'")
    val, unit = float(m.group(1)), m.group(2)
    if unit.startswith("h"):
        return val
    if unit.startswith("d"):
        return val * 24.0
    if unit.startswith("w"):
        return val * 24.0 * 7.0
    raise ValueError(f"Unknown time unit in '{tp}'")


def aggregate_scores(obs, expr, hours, tp, score_col="injury_score", comp_col="compartment"):
    """Pure-pandas aggregation (testable without scanpy).
    obs : DataFrame with columns [comp_col, score_col] indexed by spot
    expr: DataFrame spots x marker genes (log-normalised), same index"""
    g = obs.groupby(comp_col)[score_col]
    inj = pd.DataFrame({"n_spots": g.size(), "mean_score": g.mean(), "sd_score": g.std(ddof=1)})
    inj["timepoint"], inj["hours"] = tp, hours
    inj["tissue_mean"] = obs[score_col].mean()
    inj = inj.reset_index().rename(columns={comp_col: "compartment"})
    mk = expr.groupby(obs[comp_col].values).mean()
    mk = mk.stack().rename("mean_expr").reset_index()
    mk.columns = ["compartment", "gene", "mean_expr"]
    mk["timepoint"], mk["hours"] = tp, hours
    return inj, mk


def score_injury(adata, markers):
    """scanpy score_genes on the injury panel (case-insensitive gene lookup); writes adata.obs['injury_score']."""
    import scanpy as sc
    low = {v.lower(): v for v in adata.var_names}
    genes = [g if g in adata.var_names else low[g.lower()] for g in markers if (g in adata.var_names or g.lower() in low)]
    if not genes:
        raise ValueError(f"None of {markers} found in var_names")
    sc.tl.score_genes(adata, genes, score_name="injury_score")
    return genes


def main(outdir="inference_inputs"):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    import scanpy as sc
    from preprocess_compartments import SAMPLE_PATHS, SC_REFERENCE_H5AD, INJURY_MARKERS, run_pipeline
    TIMEPOINT_HOURS = {"sham": 0, "4h": 4, "12h": 12, "48h": 48, "6wk": 24 * 42}
    os.makedirs(outdir, exist_ok=True)
    parts = []
    for tp, path in SAMPLE_PATHS.items():
        done = f"{outdir}/inj_{tp}.csv"
        if os.path.exists(done):
            print(f"--- {tp}: already exported, skipping (delete {done} to redo) ---")
            parts.append(tp); continue
        print(f"--- {tp} ---")
        hours = hours_of(tp, TIMEPOINT_HOURS)
        res = run_pipeline(path, SC_REFERENCE_H5AD)
        ad = res["adata"]
        score_injury(ad, INJURY_MARKERS)                  # fills ad.obs['injury_score']
        low = {v.lower(): v for v in ad.var_names}
        genes = [low[m.lower()] for m in INJURY_MARKERS if m.lower() in low]
        X = ad[:, genes].X
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        expr = pd.DataFrame(X, index=ad.obs_names, columns=genes)
        inj, mk = aggregate_scores(ad.obs, expr, hours, tp)
        pd.DataFrame(res["W_c"], index=res["names"], columns=res["names"]).to_csv(f"{outdir}/W_c_{tp}.csv")
        pd.Series(res["names"]).to_csv(f"{outdir}/names_{tp}.csv", index=False, header=["compartment"])
        mk.to_csv(f"{outdir}/mk_{tp}.csv", index=False)
        pd.DataFrame([dict(timepoint=tp, hours=hours, kl=res.get("kl"), n_spots=ad.n_obs,
                           tissue_mean_score=float(ad.obs["injury_score"].mean()))]).to_csv(f"{outdir}/qc_{tp}.csv", index=False)
        inj.to_csv(done, index=False)                      # written last = "this timepoint is complete"
        parts.append(tp)
    pd.concat([pd.read_csv(f"{outdir}/inj_{tp}.csv") for tp in parts]).to_csv(f"{outdir}/injury_long.csv", index=False)
    pd.concat([pd.read_csv(f"{outdir}/mk_{tp}.csv") for tp in parts]).to_csv(f"{outdir}/markers_long.csv", index=False)
    pd.concat([pd.read_csv(f"{outdir}/qc_{tp}.csv") for tp in parts]).to_csv(f"{outdir}/qc.csv", index=False)
    print(f"Done. Files in {outdir}/ -> copy injury_long.csv, markers_long.csv, qc.csv, W_c_sham.csv to inputs/")


if __name__ == "__main__":
    main()
