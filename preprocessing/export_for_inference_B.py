"""
export_for_inference_B.py -- same export for GSE269622 (sham, 4 h, 12 h, day 2 -> labelled 48h), with the same
pipeline and injury score as for GSE182939.  Run from the repository root:

    python preprocessing/export_for_inference_B.py

Expected layout: data/GSE269622/extracted/mouse_<sham|hour4|hour12|day2>/outs
Output: inference_inputs_B/ -> copy injury_long.csv, markers_long.csv, qc.csv, W_c_sham.csv to inputs_B/
"""
import os, sys
import numpy as np
import pandas as pd

DATA_DIR = "data/GSE269622/extracted"
SAMPLES = {            # label used downstream : (folder, hours post-injury)
    "sham": ("mouse_sham/outs", 0.0),
    "4h":   ("mouse_hour4/outs", 4.0),
    "12h":  ("mouse_hour12/outs", 12.0),
    "48h":  ("mouse_day2/outs", 48.0),
}


def score_injury(adata, markers):
    """scanpy score_genes on the injury panel (case-insensitive gene lookup)."""
    import scanpy as sc
    low = {v.lower(): v for v in adata.var_names}
    genes = [g if g in adata.var_names else low[g.lower()] for g in markers if (g in adata.var_names or g.lower() in low)]
    if not genes:
        raise ValueError(f"None of {markers} found in var_names")
    print(f"  [injury markers] found {len(genes)}/{len(markers)}: {genes}")
    sc.tl.score_genes(adata, genes, score_name="injury_score")
    return genes


def main(outdir="inference_inputs_B"):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocess_compartments import SC_REFERENCE_H5AD, INJURY_MARKERS, run_pipeline
    from export_for_inference import aggregate_scores
    os.makedirs(outdir, exist_ok=True)
    done = []
    for tp, (sub, hours) in SAMPLES.items():
        flag = f"{outdir}/inj_{tp}.csv"
        if os.path.exists(flag):
            print(f"--- {tp}: already exported, skipping ---"); done.append(tp); continue
        path = os.path.join(DATA_DIR, sub)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{path} not found (run from the folder that contains data/)")
        print(f"--- {tp} ({path}) ---")
        res = run_pipeline(path, SC_REFERENCE_H5AD)
        ad = res["adata"]
        genes = score_injury(ad, INJURY_MARKERS)
        X = ad[:, genes].X
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        expr = pd.DataFrame(X, index=ad.obs_names, columns=genes)
        inj, mk = aggregate_scores(ad.obs, expr, hours, tp)
        pd.DataFrame(res["W_c"], index=res["names"], columns=res["names"]).to_csv(f"{outdir}/W_c_{tp}.csv")
        mk.to_csv(f"{outdir}/mk_{tp}.csv", index=False)
        pd.DataFrame([dict(timepoint=tp, hours=hours, kl=res.get("kl"), n_spots=ad.n_obs,
                           tissue_mean_score=float(ad.obs["injury_score"].mean()))]).to_csv(f"{outdir}/qc_{tp}.csv", index=False)
        inj.to_csv(flag, index=False)
        done.append(tp)
    pd.concat([pd.read_csv(f"{outdir}/inj_{t}.csv") for t in done]).to_csv(f"{outdir}/injury_long.csv", index=False)
    pd.concat([pd.read_csv(f"{outdir}/mk_{t}.csv") for t in done]).to_csv(f"{outdir}/markers_long.csv", index=False)
    pd.concat([pd.read_csv(f"{outdir}/qc_{t}.csv") for t in done]).to_csv(f"{outdir}/qc.csv", index=False)
    print(f"Done. Files in {outdir}/ -> copy injury_long.csv, markers_long.csv, qc.csv, W_c_sham.csv to inputs_B/")


if __name__ == "__main__":
    main()
