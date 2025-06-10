import os
import argparse
import re
import torch
import pandas as pd
import numpy as np
import warnings


# 1) THREADS / BLAS TUNING: ensure we’re using all CPU cores for BLAS ops
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())  # controls OpenMP threads for numpy/BLAS :contentReference[oaicite:0]{index=0}
torch.set_num_threads(os.cpu_count())               # controls PyTorch intra-op parallelism :contentReference[oaicite:1]{index=1}

# 2) SKLEARN LASSO INSTEAD OF CUSTOM ADAM LOOP
from sklearn.linear_model import Lasso as SkLasso  # coordinate descent, MKL-enabled :contentReference[oaicite:2]{index=2}
from sklearn.neural_network import MLPRegressor  # CPU‐optimized MLP :contentReference[oaicite:3]{index=3}
from scipy.stats import ttest_ind  # fast C‐backend t-test :contentReference[oaicite:4]{index=4}

# 3) PARALLELISM FOR PER-Y LASSO & dCov
from joblib import Parallel, delayed               # for easy multicore loops :contentReference[oaicite:3]{index=3}

# -------------------------
# 0. Setup
# -------------------------
device = "cpu"

def parse_feature_metadata(col: str):
    parts = col.split('__', maxsplit=3)
    if len(parts) < 3:
        raise ValueError(f"Bad column name: {col}")
    qid, name, ftype = parts[:3]
    if not re.fullmatch(r'Q\d+', qid):
        raise ValueError(f"Invalid QID in: {col}")
    return {"qid": qid, "name": name, "type": ftype}

# -------------------------
# 1. initial_screen (unchanged)
# -------------------------
def initial_screen(df, manual_seeds,
                   alpha_lin=0.05, r_thresh=0.30,
                   alpha_nl=0.05, n_resamples=200):
    from scipy.stats import pearsonr
    import dcor
    pairs = []
    for X, Y in manual_seeds:
        x = df[X].values
        y = df[Y].values
        r, p = pearsonr(x, y)
        if p < alpha_lin and abs(r) >= r_thresh:
            pairs.append((X, Y, "LIN", r))
        else:
            res = dcor.independence.distance_covariance_test(
                x.reshape(-1,1), y.reshape(-1,1),
                num_resamples=n_resamples
            )
            if res.pvalue < alpha_nl:
                pairs.append((X, Y, "NL", None))
    return pd.DataFrame(pairs, columns=["X","Y","Type","Estimate"])

# -------------------------
# 2. SEM-LASSO (approximate acyclicity by trace(W^2))
# -------------------------
def fit_sem_lasso_torch(X, pairs_df, lambda_grid, tiny_penalty,
                        max_iter=500, lr=1e-2,
                        acyc_gamma=10.0, tol=1e-4, patience=50):
    n, p = X.shape
    col_index = {col:i for i,col in enumerate(df.columns)}

    # masks for penalization
    lin_mask = torch.zeros((p,p), device=device)
    nl_mask  = torch.zeros((p,p), device=device)
    for _, r in pairs_df.iterrows():
        i,j = col_index[r["X"]], col_index[r["Y"]]
        (lin_mask if r["Type"]=="LIN" else nl_mask)[i,j] = 1.0

    best_bic = np.inf
    best_survivors = None

    for lam in lambda_grid:
        W = torch.zeros((p,p), device=device, requires_grad=True)
        opt = torch.optim.Adam([W], lr=lr)

        prev_loss = None
        wait = 0

        for it in range(max_iter):
            opt.zero_grad()
            R = X - X @ W

            # data‐fit + sparse penalties
            loss = 0.5*(R**2).sum()/n \
                   + lam*(lin_mask*W.abs()).sum() \
                   + lam*tiny_penalty*(nl_mask*W.abs()).sum()

            # cheap acyclicity: sum of squares ≈ trace(exp(W⊙W))–p :contentReference[oaicite:4]{index=4}
            h = (W*W).sum()
            loss = loss + acyc_gamma * h

            curr = loss.item()
            loss.backward()
            opt.step()

            # early‐stopping
            if prev_loss is not None and abs(prev_loss - curr) < tol:
                wait += 1
                if wait >= patience:
                    break
            else:
                wait = 0
            prev_loss = curr

        # extract survivors
        W_np = W.detach().cpu().numpy()
        survivors = [
            (r["X"], r["Y"], r["Type"], W_np[col_index[r["X"]], col_index[r["Y"]]])
            for _, r in pairs_df.iterrows()
            if abs(W_np[col_index[r["X"]], col_index[r["Y"]]])>1e-8
        ]

        mse = ((X.cpu().numpy() - X.cpu().numpy() @ W_np)**2).mean()
        k = len(survivors)
        bic = n*np.log(mse+1e-12) + k*np.log(n)
        if bic < best_bic:
            best_bic, best_survivors = bic, survivors

    return pd.DataFrame(best_survivors, columns=["X","Y","Type","Estimate"])

# -------------------------
# 3. Vectorized residual helper
# -------------------------
def get_residual_tensor(df, edges_df, Y, skip_X=None):
    """
    Compute R_Y = Y - sum_{Z in parents(Y), Z != skip_X} [beta_{Z→Y} * X[:,Z]].
    Handles numeric slopes in a vectorized batch, and piecewise (dict) slopes in a loop.
    """
    # 1) Load Y as a torch tensor
    y = torch.from_numpy(df[Y].values).float().to(device)  # torch.from_numpy supports numeric arrays only :contentReference[oaicite:2]{index=2}

    # 2) Select all parent edges for Y, excluding skip_X if given
    parents = edges_df[(edges_df["Y"] == Y)]
    if skip_X is not None:
        parents = parents[parents["X"] != skip_X]

    # 3) Handle purely numeric slopes in one batch
    #    Filter rows where Estimate is a float or int
    num = parents[parents["Estimate"].apply(lambda v: isinstance(v, (int, float)))]  # pandas apply + isinstance :contentReference[oaicite:3]{index=3}
    if not num.empty:
        cols = num["X"].tolist()
        idxs = [df.columns.get_loc(c) for c in cols]
        # Convert pandas Series of floats to a numpy float array, then to torch tensor
        slopes = torch.tensor(num["Estimate"].astype(float).values,
                              dtype=y.dtype, device=device)  # now numeric dtype, no object :contentReference[oaicite:4]{index=4}
        X_par = torch.from_numpy(df.iloc[:, idxs].values).float().to(device)
        y = y - (X_par * slopes.unsqueeze(0)).sum(dim=1)

    # 4) Handle piecewise (dict) slopes one by one
    piece = parents[parents["Estimate"].apply(lambda v: isinstance(v, dict))]
    for _, r2 in piece.iterrows():
        slope_map = r2["Estimate"]            # a dict: interval_str -> slope_val
        x2 = torch.from_numpy(df[r2["X"]].values).float().to(device)
        for interval, val in slope_map.items():
            lo, hi = map(float, interval.split(' - '))
            mask = (x2 >= lo) & (x2 <= hi)
            y[mask] -= val * x2[mask]

    return y


# -------------------------
# 4. Edge refinement (unchanged)
# -------------------------



def refine_edges(
    df,
    edges_df,
    aicc_drop=10.0,
    min_slope=0.15,
    hidden_units=4,
    mlp_lr=1e-2,
    mlp_max_iter=100,     # lowered from 200
    tol=1e-4,
    patience=5            # early stopping patience
):
    """
    Fast, parallel edge refinement with manual finite-difference slopes,
    warning suppression, and threaded joblib.
    """
    # Precompute NumPy data
    df_vals = df.values.astype(float)
    rows, _ = df_vals.shape
    cols = df.columns.tolist()
    col_idx = {c: i for i, c in enumerate(cols)}

    # Prepare tasks
    tasks = [
        (r["X"], r["Y"], r["Type"], r["Estimate"])
        for _, r in edges_df.iterrows()
    ]

    # Parallel execution
    results = Parallel(
        n_jobs=-1,
        backend="threading",
        pre_dispatch="all"
    )(
        delayed(_refine_one_edge)(
            X, Y, typ, est,
            df_vals, edges_df, col_idx, rows,
            aicc_drop, min_slope,
            hidden_units, mlp_lr,
            mlp_max_iter, tol, patience
        )
        for X, Y, typ, est in tasks
    )

    return pd.DataFrame(
        results, columns=["X","Y","Type","Estimate"]
    )

def _refine_one_edge(
    X, Y, typ, est,
    df_vals, edges_df, col_idx, rows,
    aicc_drop, min_slope,
    hidden_units, mlp_lr,
    mlp_max_iter, tol, patience
):
    # 1) Residuals (vectorized)
    y = df_vals[:, col_idx[Y]].copy()
    parents = edges_df[(edges_df["Y"]==Y) & (edges_df["X"]!=X)]
    # subtract numeric slopes
    num = parents[parents["Estimate"].apply(lambda v: isinstance(v,(int,float)))]
    if not num.empty:
        idxs = [col_idx[c] for c in num["X"]]
        slopes = np.array(num["Estimate"].tolist(), dtype=float)
        y -= (df_vals[:, idxs] * slopes).sum(axis=1)
    # subtract piecewise
    for _, pr in parents.iterrows():
        if isinstance(pr["Estimate"], dict):
            x2 = df_vals[:, col_idx[pr["X"]]]
            for interval, val in pr["Estimate"].items():
                lo, hi = map(float, interval.split(" - "))
                mask = (x2 >= lo) & (x2 <= hi)
                y[mask] -= val * x2[mask]

    R = y - y.mean()
    x = df_vals[:, col_idx[X]]

    # 2) Binary case
    meta = parse_feature_metadata(X)
    if meta["type"] == "binary":
        grp0 = R[x==0]; grp1 = R[x==1]
        _, pval = ttest_ind(grp0, grp1, equal_var=False)
        return (X, Y, "LIN", float(grp1.mean() - grp0.mean()))

    # 3) Linear fit & AIC
    xm = x - x.mean()
    beta = np.dot(xm, R) / np.dot(xm, xm)
    rss_lin = np.sum((R - beta * xm)**2)
    aic_lin = rows * np.log(rss_lin/rows + 1e-12) + 2*2

    # 4) MLP regression with early stopping
    mlp = MLPRegressor(
        hidden_layer_sizes=(hidden_units,),
        solver="adam",
        learning_rate_init=mlp_lr,
        tol=tol,
        max_iter=mlp_max_iter,
        early_stopping=True,
        n_iter_no_change=patience,
        random_state=0
    )
    mlp.fit(x.reshape(-1,1), R)
    pred = mlp.predict(x.reshape(-1,1))
    rss_gam = np.sum((R - pred)**2)
    aic_gam = rows * np.log(rss_gam/rows + 1e-12) + 2*(1 + 2*hidden_units)

    # 5) Decide LIN vs NL via finite-difference slopes
    if aic_gam + aicc_drop < aic_lin:
        # quantile midpoints
        bounds = np.quantile(x, np.linspace(0,1,5))
        mids = (bounds[:-1] + bounds[1:]) / 2
        # manual finite-difference, suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            eps = (mids[1] - mids[0]) / 2
            Xp = mids + eps; Xm = mids - eps
            pred_p = mlp.predict(Xp.reshape(-1,1))
            pred_m = mlp.predict(Xm.reshape(-1,1))
            grads = (pred_p - pred_m) / (2*eps)

        if np.max(np.abs(grads)) >= min_slope:
            slope_map = {
                f"{bounds[i]:.3f} - {bounds[i+1]:.3f}": float(grads[i])
                for i in range(len(grads))
            }
            return (X, Y, "NL", slope_map)

    # 6) Default to linear
    return (X, Y, "LIN", float(beta))


# def refine_edges(df, edges_df,
#                  aicc_drop=10.0, min_slope=0.15,
#                  mlp_iters=200, mlp_lr=1e-2,
#                  hidden_units=4, tol=1e-4, patience=20):
#     from scipy.stats import ttest_ind
#     rows, _ = df.shape
#     refined = []

#     for _, r in edges_df.iterrows():
#         X, Y, _, _ = r["X"], r["Y"], r["Type"], r["Estimate"]
#         meta = parse_feature_metadata(X)
#         y = get_residual_tensor(df, edges_df, Y, skip_X=X)
#         R = y - y.mean()
#         x = torch.from_numpy(df[X].values).float().to(device)

#         if meta["type"] == "binary":
#             grp0 = R[x==0].cpu().numpy()
#             grp1 = R[x==1].cpu().numpy()
#             _, p = ttest_ind(grp0, grp1, equal_var=False)
#             refined.append((X, Y, "LIN", grp1.mean()-grp0.mean()))
#             continue

#         # linear fit
#         xm, ym = x-x.mean(), R
#         beta = (xm*ym).sum()/(xm**2).sum()
#         rss_lin = ((ym-beta*xm)**2).sum()
#         aic_lin = rows*torch.log(rss_lin/rows+1e-12) + 2*2

#         # smaller MLP
#         net = torch.nn.Sequential(
#             torch.nn.Linear(1, hidden_units),
#             torch.nn.ReLU(),
#             torch.nn.Linear(hidden_units, 1)
#         ).to(device)
#         opt = torch.optim.Adam(net.parameters(), lr=mlp_lr)

#         prev_aic = None
#         wait = 0

#         for it in range(mlp_iters):
#             opt.zero_grad()
#             pred = net(x.unsqueeze(1)).squeeze()
#             rss_gam = ((R-pred)**2).sum()
#             aic_gam = rows*torch.log(rss_gam/rows+1e-12) + 2*(1+2*hidden_units)
#             loss = rss_gam  # train by RSS; monitor AIC
#             loss.backward()
#             opt.step()

#             # early‐stop when AIC stops improving
#             if prev_aic is not None and prev_aic - aic_gam < tol:
#                 wait += 1
#                 if wait >= patience:
#                     break
#             else:
#                 wait = 0
#             prev_aic = aic_gam

#         # final AIC & slope check
#         pred = net(x.unsqueeze(1)).squeeze()
#         rss_gam = ((R-pred)**2).sum()
#         aic_gam = rows*torch.log(rss_gam/rows+1e-12) + 2*(1+2*hidden_units)

#         if aic_gam + aicc_drop < aic_lin:
#             # piecewise slope
#             bounds = torch.quantile(x, torch.linspace(0,1,5,device=device))
#             mids = ((bounds[:-1]+bounds[1:])/2).detach().requires_grad_()
#             p_mid = net(mids.unsqueeze(1)).squeeze()
#             grads = torch.autograd.grad(p_mid.sum(), mids)[0]
#             if grads.abs().max() >= min_slope:
#                 slope_map = {
#                     f"{bounds[i].item():.3f} - {bounds[i+1].item():.3f}": grads[i].item()
#                     for i in range(len(grads))
#                 }
#                 refined.append((X, Y, "NL", slope_map))
#                 continue

#         refined.append((X, Y, "LIN", beta.item()))

#     return pd.DataFrame(refined, columns=["X","Y","Type","Estimate"])

# -------------------------
# 5. GPU‐free find_new_edges with SKLEARN Lasso & joblib
# -------------------------

def dcov_test_torch(x: torch.Tensor,
                    y: torch.Tensor,
                    n_perm: int = 200) -> tuple[float, float]:
    """
    Compute distance correlation and a permutation-based p-value in Torch.
    x, y: 1-D tensors of shape (n,), on device
    returns (dcor, pvalue)
    """
    n = x.size(0)
    # pairwise distances
    X = x.unsqueeze(1)
    Y = y.unsqueeze(1)
    a = torch.cdist(X, X, p=2)
    b = torch.cdist(Y, Y, p=2)
    # double center
    A = a - a.mean(0, keepdim=True) - a.mean(1, keepdim=True) + a.mean()
    B = b - b.mean(0, keepdim=True) - b.mean(1, keepdim=True) + b.mean()
    # dCov²
    dcov2 = (A * B).sum() / (n * n)
    dvar_x = (A * A).sum() / (n * n)
    dvar_y = (B * B).sum() / (n * n)
    dcov = torch.sqrt(dcov2.clamp(min=0.0))
    denom = torch.sqrt(dvar_x * dvar_y).clamp(min=1e-12)
    dcor = (dcov / denom).item()

    # permutation p-value
    perm_stats = torch.empty(n_perm, device=x.device)
    for i in range(n_perm):
        perm = torch.randperm(n, device=x.device)
        Bp = b[perm][:, perm]
        Bp_center = Bp - Bp.mean(0,keepdim=True) - Bp.mean(1,keepdim=True) + Bp.mean()
        dcov2_p = (A * Bp_center).sum() / (n*n)
        perm_stats[i] = torch.sqrt(dcov2_p.clamp(min=0.0))
    # count how many >= observed dcov
    pval = (perm_stats >= dcov).float().mean().item()
    return dcor, pval


def find_new_edges(df, edges_df,
                   lambda_lasso: float = 50,
                   lasso_tol: float = 1e-4,
                   lasso_thresh: float = 1e-2,
                   alpha_nl: float = 0.05,
                   n_perm: int = 200):
    """
    - Uses sklearn Lasso (coordinate descent) for R~X regressions :contentReference[oaicite:5]{index=5}
    - Parallelizes over Y using joblib Parallel :contentReference[oaicite:6]{index=6}
    """
    X_all = df.values.astype(float)
    cols = df.columns.tolist()
    idx_map = {c:i for i,c in enumerate(cols)}
    features = set(edges_df["X"]).union(edges_df["Y"])
    results = Parallel(n_jobs=-1)(
        delayed(_process_one_target)(
            Y, df, edges_df, X_all, cols, idx_map,
            lambda_lasso, lasso_tol, lasso_thresh, alpha_nl, n_perm
        )
        for Y in features
    )
    # flatten list of lists
    new = [edge for sub in results for edge in sub]
    return pd.DataFrame(new, columns=["X","Y","Type","Estimate"])


def _process_one_target(Y, df, edges_df, X_all, cols, idx_map,
                        lambda_lasso, lasso_tol, lasso_thresh, alpha_nl, n_perm):
    # residual
    R = get_residual_tensor(df, edges_df, Y, skip_X=None).cpu().numpy()
    #R = (y - y.mean()).cpu().numpy()

    # candidates
    parents = set(edges_df[edges_df["Y"]==Y]["X"])
    cands = [c for c in cols if c!=Y and c not in parents]
    if not cands:
        return []

    Xc = X_all[:, [idx_map[c] for c in cands]]

    # 1) sklearn Lasso
    lasso = SkLasso(alpha=lambda_lasso, fit_intercept=True,
                    max_iter=1000, tol=lasso_tol)
    lasso.fit(Xc, R)
    W_vals = lasso.coef_  # numpy array :contentReference[oaicite:7]{index=7}

    new_edges = []
    for w, c in zip(W_vals, cands):
        if abs(w) > lasso_thresh:
            new_edges.append((c, Y, "LIN", float(w)))

    # 2) dCov on zeros (reuse original torch-based test)
    import torch
    for w, c in zip(W_vals, cands):
        if abs(w) <= lasso_thresh:
            x = torch.from_numpy(X_all[:, idx_map[c]]).float()
            _, pval = dcov_test_torch(x, torch.from_numpy(R).float(), n_perm=n_perm)
            if pval < alpha_nl:
                new_edges.append((c, Y, "NL", None))

    return new_edges

# -------------------------
# 6. run_until_convergence & stability_filter (unchanged)
# -------------------------

def remove_bidirectional_edges(edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fast removal of 2-cycles: for each unordered pair {X,Y}, keep only
    the direction with largest |Estimate| (handles float or dict).
    """
    # Extract arrays/lists once
    Xs = edges_df["X"].values
    Ys = edges_df["Y"].values
    ests = edges_df["Estimate"].tolist()

    # 1) Compute strength for each edge
    strengths = [
        max(abs(v) for v in est.values()) if isinstance(est, dict)
        else (0.0 if est is None else abs(est))
        for est in ests
    ]

    # 2) Build unordered pair key for each edge
    pair_keys = [tuple(sorted((x, y))) for x, y in zip(Xs, Ys)]

    # 3) Attach to a temp DataFrame
    df2 = edges_df.copy()
    df2["_strength"] = strengths
    df2["_pair_key"] = pair_keys

    # 4) Sort by strength descending so strongest per pair comes first
    df2.sort_values("_strength", ascending=False, inplace=True)

    # 5) Drop duplicates on the unordered pair, keeping first (strongest)
    df2 = df2.drop_duplicates(subset="_pair_key", keep="first")

    # 6) Clean up helper columns and reset index
    return df2.drop(columns=["_strength", "_pair_key"]) \
              .reset_index(drop=True)



def run_until_convergence(df, seeds_df, lambda_grid, tiny_penalty):
    pairs = seeds_df.copy()
    iteration = 0
    while True:
        print(f"iteration: {iteration} - fitting sem lasso")
        survivors = fit_sem_lasso_torch(
            torch.from_numpy(df.values).float().to(device),
            pairs, lambda_grid, tiny_penalty
        )
        print(f"iteration: {iteration} - refining edges")
        refined = refine_edges(df, survivors)
        print(f"iteration: {iteration} - finding new edges")
        new = find_new_edges(df, refined)
        if new.empty:
            return refined
        pairs = pd.concat([refined, new], ignore_index=True) \
                  .drop_duplicates(subset=["X","Y","Type"])
        iteration += 1
        if iteration > 10:
            print("reached limit for convergence")
            return pairs


def stability_filter(df, seeds_df, lambda_grid, tiny_penalty,
                     nboot=5, frac=0.8, freq=0.9):
    from collections import Counter
    cnt = Counter()
    for i in range(nboot):
        print(f"BOOTSTRAP: {i}")
        sub = df.sample(frac=frac, replace=False)
        final = run_until_convergence(sub, seeds_df, lambda_grid, tiny_penalty)
        for _, r in final.iterrows():
            cnt[(r["X"], r["Y"], r["Type"])] += 1
    stable = [k for k,v in cnt.items() if v >= nboot*freq]
    return pd.DataFrame(stable, columns=["X","Y","Type"])

# -------------------------
# Main
# -------------------------
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="senegal_dataframe_final.xlsx")
    args = parser.parse_args()
    df = pd.read_excel(args.data, index_col=0)

    manual_seeds = [
        ("Q67__Fertilizer_Chemical__binary_nominal__1","Q0__AGR_PROD__continuous"),
        ("Q65__Variety_Imported__binary_nominal__1", "Q0__AGR_PROD__continuous"),
        ("Q69__Use_pesticide_or_herbicide__binary__1", "Q0__AGR_PROD__continuous"),
        ("Q72__Sold_VEG_Trade__binary_nominal__3", "Q0__AGR_PROD__continuous"),
        ("Q73__Machinery_daba__binary_nominal__1", "Q0__AGR_PROD__continuous"),
        ("Q73__Machinery_draft_animals_Machinery_sprayer_Machinery_daba_Machinery_hoe_sine__binary_nominal__5", "Q0__AGR_PROD__continuous"),
        ("Q73__Machinery_pump_Machinery_draft_animals_Machinery_sprayer__binary_nominal__10", "Q0__AGR_PROD__continuous"),
        ("Q73__Machinery_pump_Machinery_draft_animals_Machinery_sprayer_Machinery_daba__binary_nominal__11", "Q0__AGR_PROD__continuous"),
        ("Q73__Machinery_pump_Machinery_draft_animals_Machinery_sprayer_Machinery_daba_Machinery_hoe_sine__binary_nominal__12", "Q0__AGR_PROD__continuous"),
        ("Q73__Machinery_tractor_Machinery_cart_Machinery_daba__binary_nominal__18", "Q0__AGR_PROD__continuous"),
        ("Q74__Practice_manuring__binary_nominal__2", "Q0__AGR_PROD__continuous"),
        ("Q76__Received_agri_info_last_12m__binary__1", "Q0__AGR_PROD__continuous"),
        ("Q82__irrigation_surface__binary_nominal__3", "Q0__AGR_PROD__continuous"),
        ("Q0__Education_years__continuous", "Q0__AGR_PROD__continuous"),
        ("Q92__Current_loan_amount_XOF__continuous", "Q0__AGR_PROD__continuous"),
        ("Q0__Hope_total__continuous", "Q0__AGR_PROD__continuous")
    ]

    seeds_df = initial_screen(df, manual_seeds)
    final_edges = run_until_convergence(
        df, seeds_df,
        lambda_grid=[50,20,10,5,1,0.1],
        tiny_penalty=1e-3
    )
    print(remove_bidirectional_edges(final_edges))
    remove_bidirectional_edges(final_edges).to_csv("results.csv")

    stable = stability_filter(
        df, seeds_df,
        lambda_grid=[50,20,10,5,1,0.1],
        tiny_penalty=1e-3
    )
    final_edges = remove_bidirectional_edges(pd.merge(final_edges, stable, on=["X","Y","Type"]))
    print(final_edges)
    final_edges.to_csv("results.csv")
    

