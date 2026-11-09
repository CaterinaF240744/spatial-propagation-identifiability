import json, glob, numpy as np, re
R = "results/"
def L(f):
    try: return json.load(open(R+f))
    except: return None
def ho(v, h, coh):
    out=[]
    pat = f"result_{v}_hold{h}{'_B' if coh=='B' else ''}_s*n{'1000' if coh=='B' else '600'}.json"
    for f in sorted(glob.glob(R+pat)):
        r=json.load(open(f)); row=[x for x in r["fit_by_timepoint"] if x["held_out"]][0]
        out.append((row["rmse"], row["rmse_null_zero"], row["corr"], r["divergences"]))
    return out
T={}
for coh, tps in (("A",["4h","12h","48h","6wk"]),("B",["4h","12h","48h"])):
    print("== cohort", coh)
    for v in ["full","zone","uniform","table1","nograph"]:
        cells=[]
        for h in tps:
            o=ho(v,h,coh)
            if not o: cells.append("  --  "); continue
            rm=np.mean([x[0] for x in o]); nl=o[0][1]; sk=1-rm/nl
            T[(coh,v,h)]=dict(rmse=rm, null=nl, skill=sk, n=len(o), rng=(min(x[0] for x in o),max(x[0] for x in o)), corr=np.mean([x[2] for x in o]), div=[x[3] for x in o])
            cells.append(f"{rm:.3f}({sk:+.0%},n{len(o)})")
        print(f"{v:8s}", " ".join(cells))
json.dump({f"{k[0]}|{k[1]}|{k[2]}":v for k,v in T.items()}, open("results/holdout_summary.json","w"), indent=1, default=float)
