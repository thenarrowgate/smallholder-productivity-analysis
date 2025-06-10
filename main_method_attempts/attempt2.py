import argparse
import re
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, ttest_ind
import dcor

# -------------------------
# 0. Setup
# -------------------------
device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def parse_feature_metadata(col: str):
    parts = col.split('__', maxsplit=3)
    if len(parts) < 3:
        raise ValueError(f"Bad column name: {col}")
    qid, name, ftype = parts[:3]
    if not re.fullmatch(r'Q\d+', qid):
        raise ValueError(f"Invalid QID in: {col}")
    return {"qid": qid, "name": name, "type": ftype}

# -------------------------
# 1. EDA & Univariate Screening
# -------------------------
def initial_screen(df, manual_seeds,
                   alpha_lin=0.01, r_thresh=0.30,
                   alpha_nl=0.01, n_resamples=200):
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
# 2. SEM‐LASSO in Torch w/ Early‐Stopping
# -------------------------
def fit_sem_lasso_torch(X, pairs_df, lambda_grid, tiny_penalty,
                        max_iter=500, lr=1e-2,
                        acyc_gamma=10.0, tol=1e-4, patience=50):
    n, p = X.shape
    col_index = {col:i for i,col in enumerate(df.columns)}

    # precompute masks
    lin_mask = torch.zeros((p,p), device=device)
    nl_mask  = torch.zeros((p,p), device=device)
    for _, r in pairs_df.iterrows():
        i,j = col_index[r["X"]], col_index[r["Y"]]
        if r["Type"]=="LIN":
            lin_mask[i,j] = 1.0
        else:
            nl_mask[i,j] = 1.0

    best_bic = np.inf
    best_survivors = None

    for lam in lambda_grid:
        W = torch.zeros((p,p), device=device, requires_grad=True)
        opt = torch.optim.Adam([W], lr=lr)

        # early‐stop trackers
        prev_loss = None
        wait = 0

        for it in range(max_iter):
            opt.zero_grad()
            R = X - X @ W
            loss = 0.5*(R**2).sum()/n \
                   + lam*(lin_mask*W.abs()).sum() \
                   + lam*tiny_penalty*(nl_mask*W.abs()).sum()
            h = torch.linalg.matrix_exp(W*W).trace() - p
            loss = loss + acyc_gamma*h*h
            curr = loss.item()
            loss.backward()
            opt.step()

            # check plateau
            if prev_loss is not None and abs(prev_loss - curr) < tol:
                wait += 1
                if wait >= patience:
                    break
            else:
                wait = 0
            prev_loss = curr

        # extract survivors (same as before) …
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
# 3. Residual Helper
# -------------------------
def get_residual_tensor(df, edges_df, Y, skip_X=None):
    y = torch.from_numpy(df[Y].values).float().to(device)
    for _, r2 in edges_df[edges_df["Y"]==Y].iterrows():
        if r2["X"]==skip_X: continue
        x2 = torch.from_numpy(df[r2["X"]].values).float().to(device)
        slope = r2["Estimate"]
        if isinstance(slope, dict):
            for interval, val in slope.items():
                lo, hi = map(float, interval.split(' - '))
                mask = (x2>=lo)&(x2<=hi)
                y[mask] -= val * x2[mask]
        else:
            y -= slope * x2
    return y

# -------------------------
# 4. Edge Refinement w/ Smaller MLP & Early‐Stopping
# -------------------------
def refine_edges(df, edges_df,
                 aicc_drop=10.0, min_slope=0.1,
                 mlp_iters=200, mlp_lr=1e-2,
                 hidden_units=4, tol=1e-4, patience=20):
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
# 5. Torch-vectorized find_new_edges
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
    a = torch.cdist(X, X, p=2)                            # 
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


# Updated find_new_edges_torch
def find_new_edges_torch(df, edges_df,
                         lambda_lasso: float = 0.1,
                         lasso_iters: int = 200,
                         lasso_lr: float = 1e-2,
                         lasso_thresh: float = 1e-2,
                         alpha_nl: float = 0.001,
                         n_perm: int = 200):
    """
    Residual‐screening using one GPU‐Lasso per Y, then dCov for the leftover Xs.
    Only loops over features currently in the network (exogenous+endogenous).
    """
    X_all = torch.from_numpy(df.values).float().to(device)  # (n, p)
    cols = df.columns.tolist()
    idx_map = {c:i for i,c in enumerate(cols)}

    new = []
    # only those already in your graph
    features = set(edges_df["X"]).union(edges_df["Y"])
    for Y in features:
        # compute residual R_Y on current parents
        R = get_residual_tensor(df, edges_df, Y, skip_X=None)
        R = (R - R.mean()).to(device)

        # build candidate list: exclude Y itself and its current parents
        parents = set(edges_df[edges_df["Y"]==Y]["X"])
        cands = [c for c in cols if c!=Y and c not in parents]
        if not cands:
            continue

        # gather their data into (n, m) tensor
        idxs = [idx_map[c] for c in cands]
        Xc = X_all[:, idxs]  # (n, m)

        # 1) GPU Lasso: solve R ~ Xc with L1 penalty
        m = len(cands)
        W = torch.zeros((m,), device=device, requires_grad=True)
        opt = torch.optim.Adam([W], lr=lasso_lr)
        for _ in range(lasso_iters):
            opt.zero_grad()
            pred = Xc * W.unsqueeze(0)         # (n,m) via broadcasting
            loss = ((R.unsqueeze(1) - pred)**2).mean() \
                   + lambda_lasso * W.abs().sum()
            loss.backward()
            opt.step()

        W_vals = W.detach().cpu().numpy()
        # pick up strong linear edges
        for i, c in enumerate(cands):
            w = W_vals[i]
            if abs(w) > lasso_thresh:
                new.append((c, Y, "LIN", w))

        # 2) for the rest, run a torch‐based dCov test
        for i, c in enumerate(cands):
            if abs(W_vals[i]) > lasso_thresh:
                continue
            x = Xc[:, i]  # already a torch tensor on device
            dcor_val, pval = dcov_test_torch(x, R, n_perm=n_perm)
            if pval < alpha_nl:
                new.append((c, Y, "NL", None))

    return pd.DataFrame(new, columns=["X","Y","Type","Estimate"])


# -------------------------
# 6. Iterate until convergence
# -------------------------
def run_until_convergence(df, seeds_df, lambda_grid, tiny_penalty):
    pairs = seeds_df.copy()
    iteration = 0
    while True:
        print(f"iteration: {iteration} fitting sem lasso")
        survivors = fit_sem_lasso_torch(
            torch.from_numpy(df.values).float().to(device),
            pairs, lambda_grid, tiny_penalty
        )
        print(f"iteration: {iteration} refining edges")
        refined = refine_edges(df, survivors)
        print(f"iteration: {iteration} finding new edges")
        new = find_new_edges_torch(df, refined)
        if new.empty:
            return refined
        pairs = pd.concat([refined, new], ignore_index=True).drop_duplicates(subset=["X","Y","Type"])
        iteration += 1

# -------------------------
# 7. Stability Filter
# -------------------------
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
        ("Q1__A__continuous","Q1__B__continuous"),
        ("Q1__A__continuous","Q1__C__continuous"),
        ("Q1__B__continuous","Q1__D__binary"),
        ("Q1__D__binary",    "Q1__E__continuous"),
        ("Q1__B__continuous","Q1__F__continuous"),
        ("Q1__C__continuous","Q1__F__continuous"),
        ("Q1__A__continuous","Q1__G__ordinal"),
        ("Q1__C__continuous","Q1__E__continuous")
    ]

    seeds_df = initial_screen(df, manual_seeds)

    final_edges = run_until_convergence(
        df, seeds_df,
        lambda_grid=[50,20,10,5,1,0.1],
        tiny_penalty=1e-3
    )
    print("=== Final Edges ===")
    print(final_edges)

    stable = stability_filter(
        df, seeds_df,
        lambda_grid=[50,20,10,5,1,0.1],
        tiny_penalty=1e-3
    )
    print("=== Stable Edges ===")
    print(pd.merge(final_edges, stable, on=["X","Y","Type"]))
