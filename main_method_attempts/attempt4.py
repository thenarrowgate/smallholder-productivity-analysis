import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import re

# -------------------------
# 1. Setup GPU Device
# -------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -------------------------
# 2. Utility Functions in Torch
# -------------------------
def parse_feature_metadata(col: str):
    parts = col.split('__', maxsplit=3)
    if len(parts) not in (3,4):
        return None
    qid, name, ftype = parts[:3]
    if not re.fullmatch(r'Q\d+', qid):
        raise Exception(f"Error: column name {col} does not conform to format.")
    dummy = parts[3] if len(parts)==4 else None
    return {"qid": qid, "name": name, "type": ftype, "dummy": dummy}

def pearson_r_torch(x: torch.Tensor, y: torch.Tensor):
    xm = x - x.mean()
    ym = y - y.mean()
    return (xm * ym).sum() / torch.sqrt((xm**2).sum() * (ym**2).sum() + 1e-12)

def distance_correlation_torch(x: torch.Tensor, y: torch.Tensor):
    a = torch.cdist(x.unsqueeze(1), x.unsqueeze(1), p=2)
    b = torch.cdist(y.unsqueeze(1), y.unsqueeze(1), p=2)
    A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
    B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
    dcov = torch.sqrt((A * B).sum() / (x.size(0)**2))
    dvar_x = torch.sqrt((A * A).sum() / (x.size(0)**2))
    dvar_y = torch.sqrt((B * B).sum() / (x.size(0)**2))
    return dcov / (torch.sqrt(dvar_x * dvar_y) + 1e-12)

# -------------------------
# 3. SEM‐LASSO in Torch
# -------------------------
def fit_sem_lasso_torch(X: torch.Tensor,
                        lambda1: float = 0.1,
                        gamma: float = 10.0,
                        max_iter: int = 1000,
                        lr: float = 1e-2):
    n, p = X.shape
    W = torch.zeros((p, p), requires_grad=True, device=device)
    opt = torch.optim.Adam([W], lr=lr)
    for it in range(max_iter):
        opt.zero_grad()
        R = X - X @ W
        loss_recon = 0.5 * (R**2).sum() / n
        loss_l1 = lambda1 * W.abs().sum()
        h = torch.linalg.matrix_exp(W * W).trace() - p
        loss = loss_recon + loss_l1 + gamma * h * h
        loss.backward()
        opt.step()
        if it % 100 == 0:
            print(f"Iter {it}: loss={loss.item():.4f}, acyc={h.item():.4e}")
    return W.detach()

# -------------------------
# 4. Edge Refinement
# -------------------------
def refine_binary_edge_torch(R: torch.Tensor, X: torch.Tensor, alpha=0.01):
    x_np = X.cpu().numpy()
    r_np = R.cpu().numpy()
    grp0 = r_np[x_np==0]
    grp1 = r_np[x_np==1]
    if len(grp0) < 2 or len(grp1) < 2:
        return None, 0.0
    _, p = ttest_ind(grp0, grp1, equal_var=False)
    return p, float(grp1.mean() - grp0.mean())

def refine_continuous_edge_torch(R: torch.Tensor,
                                 X: torch.Tensor,
                                 alpha_thresh: float = -2.0,
                                 min_slope: float = 0.1,
                                 n_splines: int = 4):
    n = R.size(0)
    # Linear fit
    xm = X - X.mean()
    ym = R - R.mean()
    beta = (xm * ym).sum() / (xm**2).sum()
    resid_lin = ym - beta * xm
    ss_lin = (resid_lin**2).sum()

    # MLP surrogate
    net = torch.nn.Sequential(
        torch.nn.Linear(1, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 1)
    ).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(200):
        optim.zero_grad()
        pred = net(X.unsqueeze(1)).squeeze()
        loss = ((R - pred)**2).mean()
        loss.backward()
        optim.step()
    pred = net(X.unsqueeze(1)).squeeze()
    ss_gam = ((R - pred)**2).sum()

    # AICc proxies
    k_lin = 2
    k_gam = 1 + 8*2
    aic_lin = 2*k_lin + n * torch.log(ss_lin / n + 1e-12)
    aic_gam = 2*k_gam + n * torch.log(ss_gam / n + 1e-12)

    if aic_gam + alpha_thresh < aic_lin:
        # piecewise slopes
        boundaries = torch.quantile(X, torch.linspace(0, 1, n_splines+1, device=device))
        mids = (boundaries[:-1] + boundaries[1:]) / 2
        mids.requires_grad_()
        pred_mid = net(mids.unsqueeze(1)).squeeze()
        grads_mid = torch.autograd.grad(pred_mid.sum(), mids)[0]
        if grads_mid.abs().max() >= min_slope:
            slope_map = {
                f"{boundaries[i].item():.4f}-{boundaries[i+1].item():.4f}": grads_mid[i].item()
                for i in range(len(grads_mid))
            }
            return True, "NL", slope_map

    # if not nonlinear, only keep if beta large enough
    if abs(beta.item()) < 0.05:
        return False, None, None
    return True, "LIN", float(beta.item())

# -------------------------
# 5. Full Pipeline
# -------------------------
def main_gpu(data_path: str):
    df = pd.read_excel(data_path, index_col=0)
    cols = df.columns.tolist()
    X_np = df.values.astype(float)
    X = torch.from_numpy(X_np).float().to(device)

    # SEM‐LASSO
    W_hat = fit_sem_lasso_torch(X)

    # Extract seeds (skip identity, threshold stronger)
    seeds = []
    for i, src in enumerate(cols):
        for j, dst in enumerate(cols):
            if i == j:  # skip self-edges
                continue
            w = W_hat[i, j].item()
            if abs(w) > 0.05:  # higher threshold to reduce spurious
                seeds.append((src, dst, w))
    seeds_df = pd.DataFrame(seeds, columns=["X","Y","Estimate"])

    # Refine edges
    refined = []
    for _, r in seeds_df.iterrows():
        src, dst = r["X"], r["Y"]
        x_t = X[:, cols.index(src)]
        y_t = X[:, cols.index(dst)]
        R = y_t - y_t.mean()
        meta = parse_feature_metadata(src)
        if meta["type"] == "binary":
            p, md = refine_binary_edge_torch(R, x_t, alpha=0.001)
            if p is not None and p < 0.001 and abs(md) >= 0.05:
                refined.append((src, dst, "LIN", md))
        else:
            ok, typ, est = refine_continuous_edge_torch(R, x_t,
                                                        alpha_thresh=-2.0,
                                                        min_slope=0.1,
                                                        n_splines=4)
            if ok:
                refined.append((src, dst, typ, est))

    final_df = pd.DataFrame(refined, columns=["X","Y","Type","Estimate"])
    # drop any accidental self-edges
    final_df = final_df[final_df["X"] != final_df["Y"]].reset_index(drop=True)

    print("=== Final Edges (GPU + Piecewise Slopes) ===")
    print(final_df)
    final_df.to_csv("results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dummy.xlsx")
    args = parser.parse_args()
    main_gpu(args.data)
