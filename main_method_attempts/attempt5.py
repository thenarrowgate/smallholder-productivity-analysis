import argparse
import numpy as np
import pandas as pd
import warnings
from scipy.stats import pearsonr, ttest_ind
import dcor
from statsmodels.stats.multitest import multipletests
from semopy import Model, calc_stats
from semopy.regularization import create_regularization
from semopy.inspector import inspect
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines
import re
import torch
from scipy.stats import t as tdist

# -------------------------
# 1. Utility & Metadata
# -------------------------

def parse_feature_metadata(col: str):
    parts = col.split('__', maxsplit=3)
    if len(parts) < 3:
        raise ValueError(f"Bad column format: {col}")
    return {"qid": parts[0], "name": parts[1], "type": parts[2]}

# -------------------------
# 2. Seed Calibration
# -------------------------

def calibrate_seed_types(df: pd.DataFrame, seeds: pd.DataFrame):
    """
    For each seed (X,Y), run quick Pearson vs. GAM AICc to assign Type=LIN or NL.
    """
    types = []
    for _, row in seeds.iterrows():
        X, Y = row["X"], row["Y"]
        x = df[X].values
        y = df[Y].values
        # linear test
        try:
            r, p = pearsonr(x, y)
        except:
            r, p = 0, 1
        if p < 0.001 and abs(r) >= 0.2:
            types.append("LIN")
        else:
            # fit GAM vs OLS AICc
            beta, aic_lin, _, _ = fit_linear_model(y - y.mean(), x)
            _, aic_gam, _, _ = fit_gam_model(y - y.mean(), x)
            if aic_gam + 2 < aic_lin:
                types.append("NL")
            else:
                types.append("LIN")
    seeds["Type"] = types
    seeds["Estimate"] = pd.NA
    return seeds

# -------------------------
# 3. Pairwise Tests
# -------------------------

def pairwise_pearson_lin_sig_test(data_df, pairs_df, alpha_lin=0.001, FDR=True):
    r_vals, p_vals, indices = [], [], []
    for idx, row in pairs_df.iterrows():
        X, Y = row["X"], row["Y"]
        x, y = data_df[X].values, data_df[Y].values
        if np.all(x == x[0]) or np.all(y == y[0]):
            pairs_df.at[idx, "Type"] = "NL"
            pairs_df.at[idx, "Estimate"] = pd.NA
            continue
        r, p = pearsonr(x, y)
        if not FDR:
            if p < alpha_lin and abs(r) >= 0.1:
                pairs_df.at[idx, "Type"] = "LIN"
                pairs_df.at[idx, "Estimate"] = r
            else:
                pairs_df.at[idx, "Type"] = "NL"
                pairs_df.at[idx, "Estimate"] = pd.NA
        else:
            r_vals.append(r)
            p_vals.append(p)
            indices.append(idx)
    if FDR and p_vals:
        rej, _, _, _ = multipletests(p_vals, alpha=alpha_lin, method="fdr_bh")
        for i, idx in enumerate(indices):
            if rej[i]:
                pairs_df.at[idx, "Type"] = "LIN"
                pairs_df.at[idx, "Estimate"] = r_vals[i]
            else:
                pairs_df.at[idx, "Type"] = "NL"
                pairs_df.at[idx, "Estimate"] = pd.NA
    return pairs_df

def pairwise_dcov_nl_sig_test(data_df, pairs_df, alpha_nl=0.001, FDR=True):
    nl_idxs = pairs_df.index[pairs_df["Type"] == "NL"]
    raw, idxs, drop = [], [], []
    for idx in nl_idxs:
        X, Y = pairs_df.at[idx, "X"], pairs_df.at[idx, "Y"]
        x, y = data_df[X].values.reshape(-1,1), data_df[Y].values.reshape(-1,1)
        try:
            p, _ = dcor.independence.distance_covariance_test(x, y, num_resamples=200)
        except:
            p = 1.0
        if not FDR and p >= alpha_nl:
            drop.append(idx)
        elif FDR:
            raw.append(p); idxs.append(idx)
    if FDR and raw:
        rej, _, _, _ = multipletests(raw, alpha=alpha_nl, method="fdr_bh")
        for i, idx in enumerate(idxs):
            if not rej[i]:
                drop.append(idx)
    return pairs_df.drop(index=drop)

# -------------------------
# 4. SEM-LASSO Fit
# -------------------------

def build_sem_model(df, pairs_df):
    lines, parents = [], {}
    for _, r in pairs_df.iterrows():
        parents.setdefault(r["Y"], []).append(r["X"])
    for Y, Xs in parents.items():
        lines.append(f"{Y} ~ {' + '.join(Xs)}")
    return "\n".join(lines)

def fit_lasso_sem(df, model_desc, pairs_df, lambda_grid, tiny_penalty):
    results = []
    for lam in lambda_grid:
        mod = Model(model_desc)
        mod.load_dataset(df)
        lin = [(f"{r['Y']}~{r['X']}", lam) 
               for _,r in pairs_df[pairs_df["Type"]=="LIN"].iterrows()]
        nl  = [(f"{r['Y']}~{r['X']}", lam * tiny_penalty)
               for _,r in pairs_df[pairs_df["Type"]=="NL"].iterrows()]
        regs, grads = [], []
        for p,c in lin+nl:
            f,g = create_regularization(mod, "l1-thresh", c, [p], None)
            regs.append(f); grads.append(g)
        combined = (lambda x: sum(f(x) for f in regs),
                    lambda x: sum(g(x) for g in grads))
        try:
            mod.fit(data=df, solver="L-BFGS-B", clean_slate=True, regularization=combined)
        except:
            continue
        est = inspect(mod)
        keep = est[(est["op"]=="~") & (est["Estimate"].abs()>1e-8)]
        survivors = [(r["rval"], r["lval"]) for _,r in keep.iterrows()]
        bic = float(calc_stats(mod)["BIC"].iloc[0])
        results.append((lam, bic, survivors, mod))
    if not results:
        raise RuntimeError("No λ converged")
    best = min(results, key=lambda x: x[1])
    survivors = pd.DataFrame(best[2], columns=["X","Y"])
    return survivors, best[3]

# -------------------------
# 5. Refine Continuous/Binary
# -------------------------

def aicc(aic,k,n):
    return aic + (2*k*(k+1)) / (n-k-1) if n-k-1>0 else aic

def fit_linear_model(R, X):
    Xd = sm.add_constant(X)
    R0 = R - R.mean()
    m = OLS(R0, Xd).fit()
    return m.params[1], m.aic, m.aic, m

def fit_gam_model(R, X, n_splines=4):
    n = len(R)
    bs = BSplines(X.reshape(-1,1), df=[n_splines], degree=[3])
    gm = GLMGam(R, exog=np.ones((n,1)), smoother=bs).fit()
    aic = gm.aic
    k = gm.df_model+1
    return gm, aic, aic, bs.bs[0]

def refine_continuous_edge(R, X, alpha_threshold=-2.0, min_slope=0.1):
    beta, aic_lin, _, _ = fit_linear_model(R, X)
    gm, aic_gam, _, _ = fit_gam_model(R, X)
    if aic_gam + alpha_threshold < aic_lin:
        slopes = np.gradient(gm.predict() , X)
        if np.max(np.abs(slopes)) >= min_slope:
            # map piecewise
            return True, "NL", {"spline": slopes.tolist()}
    return True, "LIN", beta

def refine_binary_edge(R, X):
    grp0,grp1 = R[X==0], R[X==1]
    if len(grp0)<2 or len(grp1)<2:
        return None, 0.0
    _,p = ttest_ind(grp0, grp1, equal_var=False)
    return p, float(grp1.mean()-grp0.mean())

# -------------------------
# 6. Residual Screening (1 round)
# -------------------------

# def find_new_edges(df, refined, alpha_lin=0.001, alpha_nl=0.001):
#     parents = {Y: list(refined[refined["Y"]==Y]["X"]) for Y in refined["Y"].unique()}
#     new = []
#     for Y in parents:
#         # compute residual
#         R = df[Y].values.astype(float)
#         for X in parents[Y]:
#             est = refined[(refined["X"]==X)&(refined["Y"]==Y)]["Estimate"].iloc[0]
#             if refined[(refined["X"]==X)&(refined["Y"]==Y)]["Type"].iloc[0]=="LIN":
#                 R -= est * df[X].values
#         candidates = [c for c in df.columns if c!=Y and c not in parents[Y]]
#         # Pearson on residual
#         for X in candidates:
#             if pearsonr(df[X], R)[1] < alpha_lin:
#                 new.append((X,Y,"LIN",pd.NA))
#         # dCov on residual
#         for X in candidates:
#             if dcor.independence.distance_covariance_test(
#                 df[X].values.reshape(-1,1),
#                 R.reshape(-1,1), num_resamples=200
#             )[0] < alpha_nl:
#                 new.append((X,Y,"NL",pd.NA))
#     return pd.DataFrame(new, columns=["X","Y","Type","Estimate"]).drop_duplicates()

def find_new_edges(df: pd.DataFrame,
                        refined_df: pd.DataFrame,
                        alpha_lin: float = 0.001,
                        alpha_nl: float = 0.001,
                        r_min: float = 0.0):
    """
    Vectorized GPU + selective dCov version of find_new_edges:
      - df: full pandas DataFrame (n × p)
      - refined_df: DataFrame with columns [X, Y, Type, Estimate]
      - alpha_lin: significance level for Pearson
      - alpha_nl: significance level for distance-covariance
      - r_min: minimum |r| to consider (optional speedup)
    Returns a DataFrame of new edges [X, Y, Type, Estimate].
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 1) Data tensor
    X_np = df.values.astype(float)            # (n, p)
    X_all  = torch.from_numpy(X_np).float().to(device)
    cols   = df.columns.tolist()
    n, p   = X_all.shape

    # 2) Build parent map for residuals
    parents_map = {}
    for _, row in refined_df.iterrows():
        parents_map.setdefault(row.Y, []).append((row.X, row.Type, row.Estimate))

    new_edges = []

    # Precompute constants for Pearson p-value
    df_t = n - 2
    sqrt_df = np.sqrt(df_t)

    # 3) Loop over each target Y
    for j, Ycol in enumerate(cols):
        if Ycol not in parents_map:
            continue

        # 3a) Compute partial residual R = Y - sum_lin(parents)
        R = X_all[:, j].clone()
        for Xcol, typ, est in parents_map[Ycol]:
            if typ == "LIN":
                i = cols.index(Xcol)
                R -= est * X_all[:, i]
        Rc = R - R.mean()

        # 3b) Center all Xs
        Xc = X_all - X_all.mean(dim=0, keepdim=True)

        # 3c) Compute Pearson r vectorized
        num  = (Xc * Rc.unsqueeze(1)).sum(dim=0)                       # (p,)
        den  = torch.sqrt((Xc**2).sum(dim=0) * (Rc**2).sum() + 1e-12)  # (p,)
        rvec = num / den                                              # (p,)

        # 3d) Compute two‐tailed p‐values for Pearson
        r_np  = rvec.cpu().numpy()
        # avoid |r| near 1
        r_np = np.clip(r_np, -0.999999, 0.999999)
        t_stats = r_np * sqrt_df / np.sqrt(1 - r_np**2)
        pvals   = 2 * (1 - tdist.cdf(np.abs(t_stats), df_t))

        # 3e) Determine which indices pass linear screening
        lin_idx = np.where((pvals < alpha_lin) & (np.abs(r_np) >= r_min))[0]
        # Add linear edges
        for i in lin_idx:
            if i == j:
                continue
            Xcol = cols[i]
            if Xcol in [pr[0] for pr in parents_map[Ycol]]:
                continue
            new_edges.append({
                "X": Xcol,
                "Y": Ycol,
                "Type": "LIN",
                "Estimate": float(rvec[i].item())
            })

        # 3f) For non‐lin candidates, apply dCov test on CPU
        nl_candidates = set(range(p)) - set(lin_idx) - {j} \
                        - {cols.index(pr[0]) for pr in parents_map[Ycol]}
        for i in nl_candidates:
            Xcol = cols[i]
            X_block = X_all[:, i].cpu().numpy().reshape(-1,1)
            R_block = Rc.cpu().numpy().reshape(-1,1)
            p_nl, _ = dcor.independence.distance_covariance_test(
                X_block, R_block, num_resamples=200
            )
            if p_nl < alpha_nl:
                new_edges.append({
                    "X": Xcol,
                    "Y": Ycol,
                    "Type": "NL",
                    "Estimate": None
                })

    return pd.DataFrame(new_edges, columns=["X","Y","Type","Estimate"])

# -------------------------
# 7. Stability Filter
# -------------------------

def stability_filter(df, seeds, nboot=50, frac=0.8, freq=0.8, **sem_kwargs):
    from collections import Counter
    cnt, runs = Counter(), 0
    for i in range(nboot):
        sub = df.sample(frac=frac, replace=False)
        edges,_ = run_stage1(sub, seeds, **sem_kwargs)
        for _,r in edges.iterrows():
            cnt[(r["X"],r["Y"],r["Type"])] += 1
        runs += 1
    stable = [k for k,v in cnt.items() if v>=runs*freq]
    return pd.DataFrame(stable, columns=["X","Y","Type"])

# -------------------------
# 8. Parent Limiting & Pruning
# -------------------------

def prune_edges(edges, min_effect=0.05):
    return edges[edges["Estimate"].abs()>=min_effect]

def limit_parents(edges, k=3):
    out = []
    for Y,grp in edges.groupby("Y"):
        out.append(grp.reindex(grp["Estimate"].abs().sort_values(ascending=False).index).head(k))
    return pd.concat(out)

# -------------------------
# 9. Workflow Stages
# -------------------------

def run_stage1(df, seeds, lambda_grid, tiny_penalty):
    # calibrate seed types if not already
    if seeds["Type"].isna().any():
        seeds = calibrate_seed_types(df, seeds.copy())
    model_desc = build_sem_model(df, seeds)
    survivors, model = fit_lasso_sem(
        df, model_desc, seeds,
        lambda_grid=lambda_grid,
        tiny_penalty=tiny_penalty
    )
    # convert survivors to DataFrame
    survivors_df = pd.DataFrame(survivors, columns=["X","Y"])
    survivors_df["Type"] = survivors_df.apply(
        lambda r: seeds[(seeds["X"]==r["X"])&(seeds["Y"]==r["Y"])]["Type"].iloc[0],
        axis=1
    )
    survivors_df["Estimate"] = pd.NA
    # refine their estimates
    refined = []
    for _,r in survivors_df.iterrows():
        X, Y, T = r["X"], r["Y"], r["Type"]
        R = df[Y].values.astype(float)
        if T=="LIN":
            p,md = refine_binary_edge(R, df[X].values) if parse_feature_metadata(X)["type"]=="binary" else (None, None)
            if p is None or p<0.01:
                refined.append((X,Y,"LIN", md if md is not None else 0.0))
        else:
            ok, ttype, est = refine_continuous_edge(R-R.mean(), df[X].values)
            if ok:
                refined.append((X,Y,ttype, est))
    return pd.DataFrame(refined, columns=["X","Y","Type","Estimate"]), model

def run_stage2(df, edges):
    new = find_new_edges(df, edges, alpha_lin=0.001, alpha_nl=0.001)
    return pd.concat([edges, new]).drop_duplicates(subset=["X","Y","Type"])

# -------------------------
# 10. Main Script
# -------------------------

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="Fisher Information Matrix is not PD")
    np.seterr(divide='ignore', invalid='ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dummy.xlsx")
    parser.add_argument("--nboot", type=int, default=50)
    parser.add_argument("--freq", type=float, default=0.8)
    args = parser.parse_args()

    df = pd.read_excel(args.data, index_col=None)
    # define your manual seeds here (or load from file)
    manual_seeds = [
        ("Q1__A__continuous","Q1__B__continuous"),
        ("Q1__A__continuous","Q1__C__continuous"),
        ("Q1__B__continuous","Q1__D__binary"),
        ("Q1__D__binary",    "Q1__E__continuous"),
        ("Q1__B__continuous","Q1__F__continuous"),
        ("Q1__C__continuous","Q1__F__continuous"),
        ("Q1__A__continuous","Q1__G__ordinal"),
        ("Q1__C__continuous","Q1__E__continuous"),
    ]
    seeds_df = pd.DataFrame(manual_seeds, columns=["X","Y"])
    seeds_df["Type"] = pd.NA
    seeds_df["Estimate"] = pd.NA

    # Stage 1: SEM-LASSO + refine
    edges1, sem_model = run_stage1(
        df, seeds_df,
        lambda_grid=np.logspace(-3,0,20),
        tiny_penalty=1e-4
    )
    # Prune tiny effects
    edges1 = prune_edges(edges1, min_effect=0.05)
    # Limit parents per node
    edges1 = limit_parents(edges1, k=3)

    # Stage 2: one round residual screening + refine
    edges2 = run_stage2(df, edges1)
    edges2 = prune_edges(edges2, min_effect=0.05)
    edges2 = limit_parents(edges2, k=3)

    print("=== Final Sparse Edges ===")
    print(edges2)

    # Stage 3: stability filter
    stable = stability_filter(
        df, seeds_df,
        nboot=args.nboot,
        frac=0.8,
        freq=args.freq,
        lambda_grid=np.logspace(-3,0,20),
        tiny_penalty=1e-4
    )
    # Intersection
    final = pd.merge(edges2, stable, on=["X","Y","Type"])
    print("=== Stable Edges ===")
    print(final)
