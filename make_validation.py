"""make_validation.py -- summarise the validation runs.  python make_validation.py [folder_with_results]"""
import json, glob, os, sys, numpy as np
D = sys.argv[1] if len(sys.argv) > 1 else "."
M = ["PT", "DCT", "TAL", "vascular", "immune", "other"]
def J(p):
    f = glob.glob(os.path.join(D, p)) or glob.glob(os.path.join(D, "results", p))
    return json.load(open(f[0])) if f else None
for C, sfx, nn, base, holds in (("B", "_B", 1000, "result_full_B_s1n1000.json", ["4h", "12h", "48h"]),
                                ("A", "", 600, "result_full_s1n600.json", ["4h", "12h", "48h", "6wk"])):
    print(f"===== cohort {C} =====")
    print("-- prior sensitivity (full model): posterior s.d. of log beta / log rho, and P(R_PT>R_TAL)")
    for psd, f in ((0.25, f"result_full_psd0.25{sfx}_s1n{nn}.json"), (0.5, base), (1.0, f"result_full_psd1{sfx}_s1n{nn}.json")):
        r = J(f)
        if r is None: print(f"  prior sd {psd}: missing"); continue
        sb = [r["params"][f"logbeta[{m}]"]["sd"] for m in M if m != "immune"]; sr = [r["params"][f"logrho[{m}]"]["sd"] for m in M if m != "immune"]
        pt = r["P(R_PT>R_TAL)"]
        print(f"  prior sd {psd:4}: post sd logbeta {min(sb):.2f}-{max(sb):.2f}  logrho {min(sr):.2f}-{max(sr):.2f}  "
              f"ratio {min(sb+sr)/psd:.2f}-{max(sb+sr)/psd:.2f}  P(PT>TAL) prior {pt['prior']:.2f} post {pt['posterior']:.2f}  div {r['divergences']}")
    print("-- graph permutation (uniform model, held-out skill = 1 - RMSE/RMSE0)")
    for h in holds:
        real = [J(p) for p in (f"result_uniform_hold{h}{sfx}_s1n{nn}.json",)][0]
        if real is None: print(f"  {h}: real-graph run missing"); continue
        g = lambda r: [x for x in r["fit_by_timepoint"] if x["held_out"]][0]
        sk = lambda r: 1 - g(r)["rmse"] / g(r)["rmse_null_zero"]
        perms = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(D, f"result_uniform_hold{h}_perm*{sfx}_s1n{nn}.json")) + glob.glob(os.path.join(D, "results", f"result_uniform_hold{h}_perm*{sfx}_s1n{nn}.json")))]
        if not perms: print(f"  {h}: no permutations yet"); continue
        ps = np.array([sk(r) for r in perms]); s0 = sk(real)
        pval = (1 + np.sum(ps >= s0)) / (1 + len(ps))
        print(f"  {h}: real graph {s0:+.2f} | permuted mean {ps.mean():+.2f} (min {ps.min():+.2f}, max {ps.max():+.2f}, n={len(ps)}) | "
              f"P(perm >= real) = {pval:.2f} | divergences perm max {max(r['divergences'] for r in perms)}")
