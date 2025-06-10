import os
import argparse
import re
import torch
import pandas as pd
import numpy as np

# 1) THREADS / BLAS TUNING: ensure we’re using all CPU cores for BLAS ops
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())  # controls OpenMP threads for numpy/BLAS :contentReference[oaicite:0]{index=0}
torch.set_num_threads(os.cpu_count())               # controls PyTorch intra-op parallelism :contentReference[oaicite:1]{index=1}

# 2) SKLEARN LASSO INSTEAD OF CUSTOM ADAM LOOP
from sklearn.linear_model import Lasso as SkLasso  # coordinate descent, MKL-enabled :contentReference[oaicite:2]{index=2}

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
                   alpha_lin=0.01, r_thresh=0.30,
                   alpha_nl=0.01, n_resamples=200):
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
from collections import Counter

def fit_sem_lasso_torch(
    X: torch.Tensor,
    col_names: list[str],
    pairs_df: pd.DataFrame,
    lambda_grid: list[float],
    tiny_penalty: float,
    acyc_gamma: float = 10.0,
    path_frac: float = 0.5,
    max_iter: int = 500,
    lr: float = 1e-2,
    tol: float = 1e-4,
    patience: int = 50
) -> pd.DataFrame:
    n,p = X.shape
    col_index = {col: i for i, col in enumerate(col_names)}    # Precompute penalty masks
    
    lin_mask = torch.zeros((p,p), device=device)
    nl_mask  = torch.zeros((p,p), device=device)
    for _,r in pairs_df.iterrows():
        i,j = col_index[r["X"]], col_index[r["Y"]]
        (lin_mask if r["Type"]=="LIN" else nl_mask)[i,j] = 1.0

    edge_counts = Counter()
    for lam in lambda_grid:
        # --- sample reweighting (ReScore) ---
        # initial residual = X - X@W      (with W from previous iteration or zero)
        # for simplicity start W=0 each lam, then update weights by |R|
        W = torch.zeros((p,p), device=device, requires_grad=True)
        w_sample = torch.ones(n, device=device)  # uniform start
        opt = torch.optim.Adam([W], lr=lr)
        prev_loss, wait = None, 0

        for it in range(max_iter):
            opt.zero_grad()
            # weighted fit
            R = X - X @ W
            # update sample weights after a burn-in period
            if it==1:
                w_sample = (R.abs().mean(1) + 1e-6).detach()
            # apply sample weights by scaling R
            weighted_loss = (w_sample * (R**2).sum(1)).sum() / w_sample.sum()
            sparse_pen = lam*(lin_mask*W.abs()).sum() + lam*tiny_penalty*(nl_mask*W.abs()).sum()
            acyc = acyc_gamma * (W*W).sum()
            loss = weighted_loss + sparse_pen + acyc

            curr = loss.item()
            loss.backward(); opt.step()

            if prev_loss is not None and abs(prev_loss - curr) < tol:
                wait += 1
                if wait >= patience:
                    break
            else:
                wait = 0
            prev_loss = curr

        # collect survivors at this lambda
        W_np = W.detach().cpu().numpy()
        survivors = [
            (r["X"], r["Y"], r["Type"], W_np[col_index[r["X"]], col_index[r["Y"]]])
            for _,r in pairs_df.iterrows()
            if abs(W_np[col_index[r["X"]], col_index[r["Y"]]]) > 1e-8
        ]
        for e in survivors:
            edge_counts[e] += 1

    # retain only edges stable across the path
    final = [(x,y,t,w) for (x,y,t,w),c in edge_counts.items()
             if c >= path_frac * len(lambda_grid)]
    return pd.DataFrame(final, columns=["X","Y","Type","Estimate"])

def fit_sem_lasso_torch(
    X: torch.Tensor,
    col_names: list[str],
    pairs_df: pd.DataFrame,
    lambda_grid: list[float],
    tiny_penalty: float,
    max_iter: int = 500,
    lr: float = 1e-2,
    acyc_gamma: float = 10.0,
    tol: float = 1e-4,
    patience: int = 50
) -> pd.DataFrame:
    """
    X:           (n_samples, n_features) tensor
    col_names:   list of feature names, length = n_features
    pairs_df:    DataFrame with columns ["X","Y","Type","Estimate"]
    lambda_grid: list of penalty values to search
    tiny_penalty: penalty multiplier for NL edges
    """
    n, p = X.shape
    # build feature→index map from the provided column list
    col_index = {col: i for i, col in enumerate(col_names)}

    # prepare masks for LIN vs NL penalties
    lin_mask = torch.zeros((p, p), device=device)
    nl_mask  = torch.zeros((p, p), device=device)
    for _, r in pairs_df.iterrows():
        i = col_index[r["X"]]
        j = col_index[r["Y"]]
        if r["Type"] == "LIN":
            lin_mask[i, j] = 1.0
        else:
            nl_mask[i, j]  = 1.0

    best_bic = np.inf
    best_survivors = None

    for lam in lambda_grid:
        W = torch.zeros((p, p), device=device, requires_grad=True)
        opt = torch.optim.Adam([W], lr=lr)

        prev_loss = None
        wait = 0

        for _ in range(max_iter):
            opt.zero_grad()
            R = X - X @ W
            # data fit + sparse penalties
            loss = 0.5 * (R**2).sum() / n \
                   + lam * (lin_mask * W.abs()).sum() \
                   + lam * tiny_penalty * (nl_mask * W.abs()).sum()
            # approximate acyclicity penalty
            loss = loss + acyc_gamma * (W * W).sum()

            curr = loss.item()
            loss.backward()
            opt.step()

            if prev_loss is not None and abs(prev_loss - curr) < tol:
                wait += 1
                if wait >= patience:
                    break
            else:
                wait = 0
            prev_loss = curr

        # collect nonzero edges for this λ
        W_np = W.detach().cpu().numpy()
        survivors = []
        for _, r in pairs_df.iterrows():
            i = col_index[r["X"]]
            j = col_index[r["Y"]]
            w_ij = W_np[i, j]
            if abs(w_ij) > 1e-8:
                survivors.append((r["X"], r["Y"], r["Type"], w_ij))

        # compute BIC for model selection
        mse = ((X.cpu().numpy() - X.cpu().numpy() @ W_np)**2).mean()
        k   = len(survivors)
        bic = n * np.log(mse + 1e-12) + k * np.log(n)

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
def refine_edges(df, edges_df,
                 aicc_drop=10.0, min_slope=0.1,
                 mlp_iters=200, mlp_lr=1e-2,
                 hidden_units=4, tol=1e-4, patience=20):
    from scipy.stats import ttest_ind
    rows, _ = df.shape
    refined = []

    for _, r in edges_df.iterrows():
        X, Y, _, _ = r["X"], r["Y"], r["Type"], r["Estimate"]
        meta = parse_feature_metadata(X)
        y = get_residual_tensor(df, edges_df, Y, skip_X=X)
        R = y - y.mean()
        x = torch.from_numpy(df[X].values).float().to(device)

        if meta["type"] == "binary":
            grp0 = R[x==0].cpu().numpy()
            grp1 = R[x==1].cpu().numpy()
            _, p = ttest_ind(grp0, grp1, equal_var=False)
            refined.append((X, Y, "LIN", grp1.mean()-grp0.mean()))
            continue

        # linear fit
        xm, ym = x-x.mean(), R
        beta = (xm*ym).sum()/(xm**2).sum()
        rss_lin = ((ym-beta*xm)**2).sum()
        aic_lin = rows*torch.log(rss_lin/rows+1e-12) + 2*2

        # smaller MLP
        net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_units, 1)
        ).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=mlp_lr)

        prev_aic = None
        wait = 0

        for it in range(mlp_iters):
            opt.zero_grad()
            pred = net(x.unsqueeze(1)).squeeze()
            rss_gam = ((R-pred)**2).sum()
            aic_gam = rows*torch.log(rss_gam/rows+1e-12) + 2*(1+2*hidden_units)
            loss = rss_gam  # train by RSS; monitor AIC
            loss.backward()
            opt.step()

            # early‐stop when AIC stops improving
            if prev_aic is not None and prev_aic - aic_gam < tol:
                wait += 1
                if wait >= patience:
                    break
            else:
                wait = 0
            prev_aic = aic_gam

        # final AIC & slope check
        pred = net(x.unsqueeze(1)).squeeze()
        rss_gam = ((R-pred)**2).sum()
        aic_gam = rows*torch.log(rss_gam/rows+1e-12) + 2*(1+2*hidden_units)

        if aic_gam + aicc_drop < aic_lin:
            # piecewise slope
            bounds = torch.quantile(x, torch.linspace(0,1,5,device=device))
            mids = ((bounds[:-1]+bounds[1:])/2).detach().requires_grad_()
            p_mid = net(mids.unsqueeze(1)).squeeze()
            grads = torch.autograd.grad(p_mid.sum(), mids)[0]
            if grads.abs().max() >= min_slope:
                slope_map = {
                    f"{bounds[i].item():.3f} - {bounds[i+1].item():.3f}": grads[i].item()
                    for i in range(len(grads))
                }
                refined.append((X, Y, "NL", slope_map))
                continue

        refined.append((X, Y, "LIN", beta.item()))

    return pd.DataFrame(refined, columns=["X","Y","Type","Estimate"])

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
                   lambda_lasso: float = 0.1,
                   lasso_tol: float = 1e-4,
                   lasso_thresh: float = 1e-2,
                   alpha_nl: float = 0.001,
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
    y = get_residual_tensor(df, edges_df, Y, skip_X=None)
    R = (y - y.mean()).cpu().numpy()

    # candidates
    parents = set(edges_df[edges_df["Y"]==Y]["X"])
    cands = [c for c in cols if c!=Y and c not in parents]
    if not cands:
        return []

    Xc = X_all[:, [idx_map[c] for c in cands]]

    # 1) sklearn Lasso
    lasso = SkLasso(alpha=lambda_lasso, fit_intercept=False,
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
def run_until_convergence(df, seeds_df, lambda_grid, tiny_penalty):
    pairs = seeds_df.copy()
    iteration = 0
    while True:
        print(f"iteration: {iteration} - fitting sem lasso")
        survivors = fit_sem_lasso_torch(
            torch.from_numpy(df.values).float().to(device),
            df.columns.tolist(),
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

def stability_filter(df, seeds_df, lambda_grid, tiny_penalty,
                     nboot=20, frac=0.8, freq=0.7):
    from collections import Counter
    cnt = Counter()
    for _ in range(nboot):
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
    parser.add_argument("--data", default="dummy.xlsx")
    args = parser.parse_args()
    df = pd.read_excel(args.data)

    manual_seeds = [
        ("Q1__A__continuous","Q1__B__continuous")
    ]

    seeds_df = initial_screen(df, manual_seeds)
    final_edges = run_until_convergence(
        df, seeds_df,
        lambda_grid=[50,20,10,5,1,0.1],
        tiny_penalty=1e-3
    )
    print(final_edges)

    stable = stability_filter(
        df, seeds_df,
        lambda_grid=[50,20,10,5,1,0.1],
        tiny_penalty=1e-3
    )
    print(pd.merge(final_edges, stable, on=["X","Y","Type"]))
