import logging
import numpy as np
import pandas as pd
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

# ------------------------- UTILITY FUNCTIONS -------------------------

def parse_feature_metadata(col: str):
    parts = col.split('__', maxsplit=3)
    if len(parts) not in (3,4):
        return None
    qid, name, ftype = parts[:3]
    if not re.fullmatch(r'Q\d+', qid):
        raise Exception(f"Error: column name {col} does not conform to format.")
    dummy = parts[3] if len(parts)==4 else None
    return {"qid": qid, "name": name, "type": ftype, "dummy": dummy}

# ------------------------- PAIRWISE TESTS -------------------------

def pairwise_pearson_lin_sig_test(
    data_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    alpha_lin: float = 0.05,
    FDR=True
) -> pd.DataFrame:
    """
    For each (X,Y) in pairs_df, compute Pearson correlation on data_df[X], data_df[Y].
    If FDR=False: assign LIN if p<alpha_lin and |r|>=0.1 (weaker threshold), else NL.
    If FDR=True: collect all p's, run BH-FDR, then assign LIN if adjusted p<alpha_lin, else NL.
    """
    r_vals = []
    p_vals = []

    for idx, row in pairs_df.iterrows():
        X = row["X"]
        Y = row["Y"]
        if Y not in data_df.columns or X not in data_df.columns:
            pairs_df.at[idx, "Type"] = "NL"
            pairs_df.at[idx, "Estimate"] = pd.NA
            continue

        x_vals = data_df[X].values
        y_vals = data_df[Y].values
        # Skip constants
        if np.all(x_vals == x_vals[0]) or np.all(y_vals == y_vals[0]):
            pairs_df.at[idx, "Type"] = "NL"
            pairs_df.at[idx, "Estimate"] = pd.NA
            continue

        try:
            r, p = pearsonr(x_vals, y_vals)
        except Exception:
            r, p = 0.0, 1.0
        # record
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

    if FDR:
        if len(p_vals) > 0:
            reject_mask, pvals_fdr, _, _ = multipletests(p_vals, alpha=alpha_lin, method="fdr_bh")
            for i, idx in enumerate(pairs_df.index):
                if idx < len(p_vals):  # align indices
                    if reject_mask[i]:
                        pairs_df.at[idx, "Type"] = "LIN"
                        pairs_df.at[idx, "Estimate"] = r_vals[i]
                    else:
                        pairs_df.at[idx, "Type"] = "NL"
                        pairs_df.at[idx, "Estimate"] = pd.NA
    return pairs_df


def pairwise_dcov_nl_sig_test(
    data_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    alpha_nl: float = 0.05,
    FDR=True
) -> pd.DataFrame:
    """
    For each row with Type=='NL', run distance covariance test. If FDR=False: drop if p>=alpha_nl.
    If FDR=True: collect p's across NL rows, run BH-FDR, keep those with adjusted p<alpha_nl.
    """
    nl_indices = pairs_df.index[pairs_df["Type"] == "NL"].tolist()
    if not nl_indices:
        return pairs_df
    raw_pvals = []
    to_drop = []
    for idx in nl_indices:
        X = pairs_df.at[idx, "X"]
        Y = pairs_df.at[idx, "Y"]
        X_block = data_df[X].values.reshape(-1,1)
        Y_block = data_df[Y].values.reshape(-1,1)
        try:
            p, _ = dcor.independence.distance_covariance_test(X_block, Y_block, num_resamples=200)
        except Exception:
            p = 1.0
        if not FDR:
            if p >= alpha_nl:
                to_drop.append(idx)
        else:
            raw_pvals.append((idx, p))
    if FDR and raw_pvals:
        idxs, pvals = zip(*raw_pvals)
        reject_mask, pvals_fdr, _, _ = multipletests(pvals, alpha=alpha_nl, method="fdr_bh")
        for i, idx in enumerate(idxs):
            if not reject_mask[i]:
                to_drop.append(idx)
    if to_drop:
        pairs_df = pairs_df.drop(index=to_drop)
    return pairs_df

# ------------------------- SEM BUILD & FIT -------------------------

def build_sem_model(df: pd.DataFrame, pairs_df: pd.DataFrame) -> str:
    lines = []
    parents_of = {}
    for _, row in pairs_df.iterrows():
        X = row["X"]
        Y = row["Y"]
        parents_of.setdefault(Y, []).append(X)
    for Y, X_list in parents_of.items():
        rhs = " + ".join(X_list)
        lines.append(f"{Y} ~ {rhs}")
    return "\n".join(lines)


def fit_lasso_sem(
    df: pd.DataFrame,
    model_desc: str,
    pairs_df: pd.DataFrame,
    lambda_grid=[1.0, 0.5, 0.1],
    tiny_penalty=1e-2,
    alpha_reg="l1-thresh",
    fdr_alpha=0.05
):
    np.int = int  # compatibility
    lin_seeds = pairs_df.loc[pairs_df["Type"] == "LIN", ["X", "Y"]].itertuples(index=False, name=None)
    nl_seeds  = pairs_df.loc[pairs_df["Type"] == "NL", ["X", "Y"]].itertuples(index=False, name=None)
    param_names_lin = [f"{Y}~{X}" for (X, Y) in lin_seeds]
    param_names_nl  = [f"{Y}~{X}" for (X, Y) in nl_seeds]
    results = []
    for lam in lambda_grid:
        mod = Model(model_desc)
        mod.load_dataset(df)
        regu_lin, regu_lin_grad = None, None
        regu_nl, regu_nl_grad = None, None
        if param_names_lin:
            regu_lin, regu_lin_grad = create_regularization(
                model=mod,
                regularization=alpha_reg,
                c=lam * 1.0,
                param_names=param_names_lin,
                mx_names=None
            )
        if param_names_nl:
            regu_nl, regu_nl_grad = create_regularization(
                model=mod,
                regularization=alpha_reg,
                c=lam * tiny_penalty,
                param_names=param_names_nl,
                mx_names=None
            )
        if regu_lin is None and regu_nl is None:
            combined_regu = None
        elif regu_lin is None:
            combined_regu = (regu_nl, regu_nl_grad)
        elif regu_nl is None:
            combined_regu = (regu_lin, regu_lin_grad)
        else:
            f1, g1 = regu_lin, regu_lin_grad
            f2, g2 = regu_nl,  regu_nl_grad
            combined_regu = (
                lambda x: f1(x) + f2(x),
                (None if g1 is None or g2 is None else (lambda x: g1(x) + g2(x)))
            )
        try:
            mod.fit(
                data=df,
                solver="L-BFGS-B",
                clean_slate=True,
                regularization=combined_regu
            )
        except Exception:
            continue
        est_df = inspect(mod)
        regs = est_df[est_df["op"] == "~"].copy()
        nonzero = regs[np.abs(regs["Estimate"]) > 1e-8]
        survivors = nonzero.apply(lambda row: f"{row['lval']}~{row['rval']}", axis=1).tolist()
        stats = calc_stats(mod)
        bic = float(stats["BIC"].iloc[0])
        results.append((lam, bic, survivors, mod))
    if not results:
        raise RuntimeError("No λ in lambda_grid converged successfully.")
    results.sort(key=lambda tup: tup[1])
    best_lambda, best_bic, best_survivors, best_model = results[0]
    est_final = inspect(best_model)
    regs_final = est_final[est_final["op"] == "~"].copy()
    regs_final["X"] = regs_final["rval"]
    regs_final["Y"] = regs_final["lval"]
    lin_set = set(pairs_df.loc[pairs_df["Type"] == "LIN", :].apply(lambda r: (r["X"], r["Y"]), axis=1).tolist())
    nl_set = set(pairs_df.loc[pairs_df["Type"] == "NL", :].apply(lambda r: (r["X"], r["Y"]), axis=1).tolist())
    regs_final["Type"] = regs_final.apply(
        lambda row: "LIN" if (row["X"], row["Y"]) in lin_set else "NL", axis=1
    )
    final_survivors = regs_final[np.abs(regs_final["Estimate"]) > 1e-8].copy()
    surviving_edges = final_survivors[["X", "Y", "Estimate", "Type"]].reset_index(drop=True)
    stats_final = calc_stats(best_model)
    obs_vars = best_model.vars['observed']
    Xmat = df[obs_vars].values
    S = np.cov(Xmat, rowvar=False, bias=False)
    Sigma, _ = best_model.calc_sigma()
    S_corr = cov_to_corr(S)
    Sigma_corr = cov_to_corr(Sigma)
    residuals = S_corr - Sigma_corr
    srmr = np.sqrt(np.mean(residuals[np.triu_indices_from(residuals, k=1)] ** 2))
    fit_indices = {
        "AIC":   float(stats_final["AIC"].iloc[0]),
        "BIC":   float(stats_final["BIC"].iloc[0]),
        "CFI":   float(stats_final["CFI"].iloc[0]),
        "RMSEA": float(stats_final["RMSEA"].iloc[0]),
        "SRMR":  srmr
    }
    return best_lambda, surviving_edges, fit_indices

# Convert covariance to correlation

def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    stddev = np.sqrt(np.diag(cov))
    corr = cov / np.outer(stddev, stddev)
    return corr

# ------------------------- PARTIAL RESIDUAL & REFINEMENT -------------------------

def compute_partial_residual(Y, df, parents_of, coeff_info, X=None):
    y_vec = df[Y].values.astype(float).copy()
    for Z in parents_of.get(Y, []):
        if Z == X:
            continue
        est_info = coeff_info[(Z, Y)]
        z_vec = df[Z].values.astype(float)
        if isinstance(est_info, (float, int)):
            y_vec -= float(est_info) * z_vec
        else:
            z_effect = np.zeros_like(z_vec, dtype=float)
            # est_info is a dict of {"lo - hi": slope}
            for interval_str, slope_val in est_info.items():
                lo_str, hi_str = interval_str.split(" - ")
                lo, hi = float(lo_str), float(hi_str)
                mask = (z_vec >= lo) & (z_vec <= hi)
                z_effect[mask] = slope_val * z_vec[mask]
            y_vec -= z_effect
    return y_vec


def aicc_from_aic(aic: float, k: int, n: int) -> float:
    if n - k - 1 <= 0:
        return aic
    return aic + (2.0 * k * (k + 1)) / float(n - k - 1)


def fit_linear_model(R: np.ndarray, X: np.ndarray):
    n = len(R)
    X_design = sm.add_constant(X)
    R_centered = R - R.mean()
    model = OLS(R_centered, X_design).fit()
    beta_hat = model.params[1]
    aic = model.aic
    k = int(model.df_model) + 1
    aicc_lin = aicc_from_aic(aic, k, n)
    p_val = float(model.pvalues[1])
    return beta_hat, aicc_lin, p_val, model


def fit_gam_model(
    R: np.ndarray,
    X: np.ndarray,
    n_splines: int = 4
):
    n = len(R)
    x_min, x_max = np.min(X), np.max(X)
    x_1d = X.reshape(-1, 1)
    bs = BSplines(x_1d, df=[n_splines], degree=[3])
    gam_model = GLMGam(R, exog=np.ones((n, 1)), smoother=bs)
    gam_res = gam_model.fit()
    aic = float(gam_res.aic)
    k = int(gam_res.df_model) + 1
    aicc_gam = aic if n - k - 1 <= 0 else aic + (2.0 * k * (k + 1)) / float(n - k - 1)
    cut_probs = np.linspace(0.0, 1.0, num=n_splines)
    boundaries = np.quantile(X, cut_probs)
    interval_slopes = np.empty(n_splines - 1, dtype=float)
    eps = 1e-6 * (x_max - x_min + 1e-6)
    for i in range(n_splines - 1):
        b_lo = boundaries[i] + eps
        b_hi = boundaries[i + 1] - eps
        x_pred = np.array([b_lo, b_hi]).reshape(-1, 1)
        exog_pred = np.ones((2, 1))
        preds = gam_res.predict(exog=exog_pred, exog_smooth=x_pred)
        interval_slopes[i] = (preds[1] - preds[0]) / (b_hi - b_lo)
    return gam_res, aicc_gam, boundaries, interval_slopes


def refine_continuous_edge(R: np.ndarray, 
                           X: np.ndarray,
                           alpha_threshold: float = 2.0  # force GAM to beat linear by ≥2 AICc
                          ):
    beta_lin, aicc_lin, p_lin, _ = fit_linear_model(R, X)
    gam_res, aicc_gam, boundaries, knot_slopes = fit_gam_model(R, X)

    # Require GAM AICc to be at least 2 points smaller than linear’s AICc
    if aicc_gam + alpha_threshold < aicc_lin:
        MIN_SLOPE = 0.15  # require a strong nonlinear slope
        if any(abs(s) >= MIN_SLOPE for s in knot_slopes):
            return True, "NL", {f"{boundaries[i]} - {boundaries[i+1]}": knot_slopes[i]
                                 for i in range(len(knot_slopes))}
    return True, "LIN", beta_lin


def refine_binary_edge(R: np.ndarray, X: np.ndarray):
    grp0 = R[X == 0]
    grp1 = R[X == 1]
    if len(grp0) < 2 or len(grp1) < 2:
        return None, 0.0
    tstat, p_val = ttest_ind(grp0, grp1, equal_var=False)
    mean_diff = float(np.mean(grp1) - np.mean(grp0))
    return p_val, mean_diff


def build_parent_map_and_coefficients(refined_edges):
    parents_of = {}
    coeff_info = {}
    for _, row in refined_edges.iterrows():
        X, Y, typ, est = row["X"], row["Y"], row["Type"], row["Estimate"]
        parents_of.setdefault(Y, []).append(X)
        coeff_info[(X, Y)] = est
    return parents_of, coeff_info


def refine_edges(df: pd.DataFrame, surviving_edges: pd.DataFrame):
    parents_of, coefs = build_parent_map_and_coefficients(surviving_edges)
    cont_info = []
    refined = []
    for _, row in surviving_edges.iterrows():
        X, Y, typ, est_val = row["X"], row["Y"], row["Type"], row["Estimate"]
        R_Y = compute_partial_residual(Y, df, parents_of, coefs, X)
        R_Y_centered = R_Y - R_Y.mean()
        meta = parse_feature_metadata(X)
        if meta["type"] in ["continuous", "ordinal"]:
            cont_info.append((X, Y, R_Y_centered, df[X].values.astype(float), typ, est_val))
        elif meta["type"] == "binary":
            refined.append((X, Y, "LIN", est_val))
        else:
            continue
    for X, Y, R_Y_centered, X_vals, typ, est_val in cont_info:
        if typ == "LIN":
            refined.append((X, Y, "LIN", est_val))
        else:
            keep, ttype, est = refine_continuous_edge(R_Y_centered, X_vals)
            if keep:
                refined.append((X, Y, ttype, est))
    df_refined = pd.DataFrame(refined, columns=["X", "Y", "Type", "Estimate"])
    return df_refined

# ------------------------- RESIDUAL SCREENING -------------------------

def find_new_edges(
    df: pd.DataFrame,
    refined_edges: pd.DataFrame,
    alpha_lin: float = 0.05,
    alpha_nl: float = 0.05
) -> pd.DataFrame:
    parents_of, coeff_info = build_parent_map_and_coefficients(refined_edges)
    all_features = list(df.columns)
    new_edges = []
    for Y in refined_edges["Y"].unique():
        R_Y = compute_partial_residual(Y, df, parents_of, coeff_info)
        candidates = [Xp for Xp in all_features if Xp != Y and Xp not in parents_of.get(Y, [])]
        if not candidates:
            continue
        pairs_lin = pd.DataFrame({"X": candidates, "Y": ["R_Y"]*len(candidates)})
        pairs_lin["Type"] = pd.NA
        pairs_lin["Estimate"] = pd.NA
        df_with_R = df.assign(R_Y=R_Y)
        pairs_lin = pairwise_pearson_lin_sig_test(
            data_df=df_with_R,
            pairs_df=pairs_lin,
            alpha_lin=alpha_lin,
            FDR=True
        )
        lin_rows = pairs_lin[pairs_lin["Type"] == "LIN"]
        for _, row in lin_rows.iterrows():
            Xp = row["X"]
            new_edges.append((Xp, Y, "LIN", pd.NA))
        nl_candidates = pairs_lin[pairs_lin["Type"] == "NL"]["X"].tolist()
        if not nl_candidates:
            continue
        pairs_nl = pd.DataFrame({"X": nl_candidates, "Y": ["R_Y"]*len(nl_candidates)})
        pairs_nl["Type"] = "NL"
        pairs_nl["Estimate"] = pd.NA
        pairs_nl = pairwise_dcov_nl_sig_test(
            data_df=df_with_R,
            pairs_df=pairs_nl,
            alpha_nl=alpha_nl,
            FDR=True
        )
        for _, row in pairs_nl.iterrows():
            Xp = row["X"]
            new_edges.append((Xp, Y, "NL", pd.NA))
    if new_edges:
        return pd.DataFrame(new_edges, columns=["X", "Y", "Type", "Estimate"]).drop_duplicates()
    else:
        return pd.DataFrame(columns=["X", "Y", "Type", "Estimate"])

# ------------------------- ITERATION & STABILITY -------------------------

def run_until_convergence(df, pairs_df_initial,
                          lambda_grid=[1.0, 0.5, 0.1],
                          tiny_penalty=1e-2,
                          alpha_lin=0.05,
                          alpha_nl=0.05):
    pairs_df = pairs_df_initial.copy().reset_index(drop=True)
    converged = False
    iteration = 0
    while not converged:
        print(f"Iteration: {iteration}")
        print("building SEM model")
        model_desc = build_sem_model(df, pairs_df)
        print("fitting lasso sem")
        best_lambda, survivors, fit_indices = fit_lasso_sem(
            df=df,
            model_desc=model_desc,
            pairs_df=pairs_df,
            lambda_grid=lambda_grid,
            tiny_penalty=tiny_penalty,
            alpha_reg="l1-thresh",
            fdr_alpha=alpha_lin
        )
        print("refining edges")
        refined = refine_edges(df, survivors)
        print("finding new edges")
        new_edges = find_new_edges(df, refined,
                                   alpha_lin=alpha_lin,
                                   alpha_nl=alpha_nl)
        if new_edges.empty:
            converged = True
            final_edges = refined.copy().reset_index(drop=True)
            break
        pairs_df = pd.concat([refined, new_edges], ignore_index=True).drop_duplicates(subset=["X","Y","Type"])
        iteration += 1
    return final_edges, fit_indices


def stability_filter(
    df: pd.DataFrame,
    pairs_df_initial: pd.DataFrame,
    nboot: int = 20,
    sample_frac: float = 0.8,
    lambda_grid=[1.0, 0.5, 0.1],
    tiny_penalty=1e-2,
    alpha_lin=0.05,
    alpha_nl=0.05,
    freq_threshold: float = 0.7
):
    from collections import Counter
    freq = Counter()
    n = df.shape[0]
    subsample_size = int(sample_frac * n)
    successful_runs = 0
    for i in range(nboot):
        subsample_idx = np.random.choice(df.index, size=subsample_size, replace=False)
        df_sub = df.loc[subsample_idx].reset_index(drop=True)
        print(f"BOOTSTRAP: {i}")
        try:
            final_sub, _ = run_until_convergence(
                df=df_sub,
                pairs_df_initial=pairs_df_initial,
                lambda_grid=lambda_grid,
                tiny_penalty=tiny_penalty,
                alpha_lin=alpha_lin,
                alpha_nl=alpha_nl
            )
            successful_runs += 1
        except Exception:
            continue
        for _, row in final_sub.iterrows():
            freq[(row["X"], row["Y"], row["Type"])] += 1
    min_count = int(np.ceil(freq_threshold * successful_runs))
    stable_edges = [triple for triple, count in freq.items() if count >= min_count]
    return stable_edges

# ------------------------- MAIN ENTRYPOINT -------------------------

def main_with_manual_seeds():
    # 1) Load the dummy data
    df = pd.read_excel("dummy.xlsx", index_col=None)

    # 2) Choose our 8 seeds, specifying exact LIN vs NL
    manual_seeds = [
        ("Q1__A__continuous", "Q1__B__continuous"),
        ("Q1__A__continuous", "Q1__C__continuous"),
        ("Q1__B__continuous", "Q1__D__binary"),
        ("Q1__D__binary",     "Q1__E__continuous"),
        ("Q1__B__continuous", "Q1__F__continuous"),
        ("Q1__C__continuous", "Q1__F__continuous"),
        ("Q1__A__continuous", "Q1__G__ordinal"),
        ("Q1__C__continuous", "Q1__E__continuous")
    ]
    seed_types = {
        ("Q1__A__continuous","Q1__B__continuous"): "LIN",
        ("Q1__A__continuous","Q1__C__continuous"): "NL",
        ("Q1__B__continuous","Q1__D__binary")   : "LIN",
        ("Q1__D__binary",    "Q1__E__continuous"): "NL",
        ("Q1__B__continuous","Q1__F__continuous"): "LIN",
        ("Q1__C__continuous","Q1__F__continuous"): "NL",
        ("Q1__A__continuous","Q1__G__ordinal")   : "NL",
        ("Q1__C__continuous","Q1__E__continuous"): "NL",
    }
    initial_pairs_df = pd.DataFrame(manual_seeds, columns=["X","Y"])\
                         .drop_duplicates().reset_index(drop=True)
    initial_pairs_df["Type"] = initial_pairs_df.apply(
        lambda r: seed_types[(r["X"], r["Y"])], axis=1
    )
    initial_pairs_df["Estimate"] = pd.NA

    print("=== Seeds (with true LIN/NL) ===")
    print(initial_pairs_df)

    # 3) Run SEM‐LASSO / refine / residual‐screen until convergence
    final_edges, fit_indices = run_until_convergence(
        df=df,
        pairs_df_initial=initial_pairs_df,
        # << lighter LASSO penalties >>
        lambda_grid=[0.1, 0.05, 0.01],
        tiny_penalty=1e-4,
        # << stricter thresholds in refine() >>
        alpha_lin=0.0005,
        alpha_nl=0.0005
    )

    print("=== Final Edges (Dummy) ===")
    print(final_edges)
    print("=== Fit Indices ===")
    print(fit_indices)

    # 4) Stability‐filter (bootstrapping)
    stable_list = stability_filter(
        df=df,
        pairs_df_initial=initial_pairs_df,
        nboot=50,
        sample_frac=0.9,
        lambda_grid=[0.1, 0.05, 0.01],
        tiny_penalty=1e-4,
        alpha_lin=0.0005,
        alpha_nl=0.0005,
        freq_threshold=0.9
    )
    stable_df = pd.DataFrame(stable_list, columns=["X","Y","Type"]).drop_duplicates()

    # 5) Merge “final_edges” ∧ “stable_edges”
    merged = pd.merge(final_edges, stable_df, on=["X","Y","Type"])
    print("=== Stable Edges (Intersect) ===")
    print(merged)
    merged.to_csv("results.csv")

if __name__ == "__main__":
    main_with_manual_seeds()
