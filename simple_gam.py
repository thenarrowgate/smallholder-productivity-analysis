#%%
import pandas as pd
import numpy as np
from pygam import LinearGAM, s
from collections import defaultdict
# At top of your file, make sure you have:
import matplotlib.pyplot as plt
from IPython import display

# (and the scipy patch you already have)
import scipy.sparse
scipy.sparse.csr_matrix.A = property(lambda self: self.toarray())
#%%
# 1) Load your data
nepal_df   = pd.read_excel("nepal_dataframe_FA.xlsx", index_col=0)
senegal_df = pd.read_excel("senegal_dataframe_FA.xlsx", index_col=0)
#%%
# 2) Function to fit a GAM and return avg derivative per X‐interval
def fit_gam_avg_derivative(df, x_col, y_col, n_grid=50, spline_k=20):
    # drop NA
    sub = df[[x_col, y_col]].dropna()
    X = sub[x_col].values.reshape(-1,1)
    y = sub[y_col].values

    # fit LinearGAM with one spline term
    gam = LinearGAM(s(0, n_splines=spline_k)).fit(X, y)

    # grid over the observed range of X
    x_min, x_max = X.min(), X.max()
    grid = np.linspace(x_min, x_max, n_grid)
    yhat = gam.predict(grid)
    
    # finite‐difference derivative
    dx    = np.diff(grid)
    dy    = np.diff(yhat)
    deriv = dy/dx

    # package into a DataFrame
    out = pd.DataFrame({
      'interval_start': grid[:-1],
      'interval_end'  : grid[1:],
      'avg_derivative': deriv
    })
    out.attrs['x_col'] = x_col
    out.attrs['y_col'] = y_col
    return out
#%%
# 3) Nepal: list of (X,Y) pairs
nepal_pairs = [
    ("Q0__hope_total__continuous",       "Q0__AGR_PROD__continuous"),
    ("Q0__self_control_score__continuous","Q0__AGR_PROD__continuous"),
    ("Q0__average_of_farming_practices__ordinal","Q0__AGR_PROD__continuous"),
    ("Q0__Positive_Negative_Score__continuous",  "Q0__AGR_PROD__continuous"),
]

nepal_results = {}
for Xcol, Ycol in nepal_pairs:
    nepal_results[(Xcol,Ycol)] = fit_gam_avg_derivative(
        nepal_df, Xcol, Ycol, n_grid=100, spline_k=25
    )

# Example: view the first few rows for the first pair
print(nepal_results[nepal_pairs[0]].head())

#%%
# 4) Senegal: first build Q0_Avg_practices
# Assume Q74__Practice__nominal holds Python lists or comma‐sep strings
def parse_list_cell(cell):
    if pd.isna(cell): 
        return []
    if isinstance(cell, list):
        return cell
    # if it's a string repr like "['A','B']" or "A;B"
    try:
        # try literal eval
        from ast import literal_eval
        return list(literal_eval(cell))
    except:
        # fallback: split on semicolon or comma
        return [s.strip() for s in str(cell).replace(";",",").split(",") if s.strip()]
#%%
# explode all practice values to find the universe
all_pracs = set()
senegal_df["Q74__Practice__nominal"].dropna().map(parse_list_cell).map(
    lambda L: all_pracs.update(L)
)
max_practices = len(all_pracs)

# compute ratio
senegal_df["Q0_Avg_practices"] = (
    senegal_df["Q74__Practice__nominal"]
      .map(parse_list_cell)
      .map(lambda L: len(L)/max_practices if max_practices>0 else np.nan)
)
#%%
# define the pairs
senegal_pairs = [
    ("Q0__Hope_total__continuous", "Q0__AGR_PROD__continuous"),
    ("Q0__Average_CS__continuous","Q0__AGR_PROD__continuous"),
    ("Q0_Avg_practices",           "Q0__AGR_PROD__continuous"),
]

senegal_results = {}
for Xcol, Ycol in senegal_pairs:
    senegal_results[(Xcol,Ycol)] = fit_gam_avg_derivative(
        senegal_df, Xcol, Ycol, n_grid=100, spline_k=25
    )
#%%
# Example: view the results for Q0_Avg_practices
print(senegal_results[("Q0_Avg_practices","Q0__AGR_PROD__continuous")].head())
# %%
def plot_gam_with_intervals(df, x_col, y_col, n_grid=100, spline_k=25):
    """
    Fit a univariate GAM of y_col ~ s(x_col), plot data + smooth,
    draw vertical lines at each interval, and annotate avg derivative.
    Returns the DataFrame of intervals + avg_derivative.
    """
    # 1. Prepare data
    sub = df[[x_col, y_col]].dropna()
    X   = sub[x_col].values.reshape(-1,1)
    y   = sub[y_col].values

    # 2. Fit
    gam = LinearGAM(s(0, n_splines=spline_k)).fit(X, y)

    # 3. Create grid & predict
    grid = np.linspace(X.min(), X.max(), n_grid)
    yhat = gam.predict(grid)

    # 4. Compute finite-difference derivative
    dx    = np.diff(grid)
    dy    = np.diff(yhat)
    deriv = dy / dx

    # 5. Build intervals table
    intervals = pd.DataFrame({
      'interval_start': grid[:-1],
      'interval_end'  : grid[1:],
      'avg_derivative': deriv
    })

    # 6. Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(X, y, alpha=0.3, s=20, label='data')
    ax.plot(grid, yhat, 'r-', lw=2, label='GAM fit')

    # vertical lines + text
    y_max = ax.get_ylim()[1]
    for _, row in intervals.iterrows():
        start, end, avg = row
        ax.axvline(start, color='gray', linestyle='--', linewidth=0.8)
        mid = (start + end) / 2
        ax.text(mid, y_max * 0.95,
                f"{avg:.2f}",
                ha='center', va='top',
                fontsize=8, rotation=90)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"GAM: {y_col} ~ s({x_col})")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return intervals
# %%
for Xcol, Ycol in nepal_pairs:
    print("Nepal:", Xcol, "→", Ycol)
    ints = plot_gam_with_intervals(
        nepal_df, Xcol, Ycol,
        n_grid=100, spline_k=25
    )
    display(ints.head())     # if in a notebook, else print(ints)
# %%
#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s
import scipy.sparse

# Monkey‐patch so pygam can call .A on csr_matrix
scipy.sparse.csr_matrix.A = property(lambda self: self.toarray())

#%% load data
nepal_df   = pd.read_excel("nepal_dataframe_FA.xlsx", index_col=0)
senegal_df = pd.read_excel("senegal_dataframe_FA.xlsx", index_col=0)

#%% helper: fit GAM, compute avg derivative per fixed‐width bin, and plot
def plot_gam_with_bins(df, x_col, y_col, n_bins=10, n_grid=200, spline_k=25):
    sub = df[[x_col, y_col]].dropna()
    X   = sub[x_col].values.reshape(-1,1)
    y   = sub[y_col].values

    # 1) fit
    gam = LinearGAM(s(0, n_splines=spline_k)).fit(X, y)

    # 2) fine grid for smooth + derivative
    grid = np.linspace(X.min(), X.max(), n_grid)
    yhat = gam.predict(grid)
    deriv = np.diff(yhat) / np.diff(grid)

    # 3) define bin edges and compute avg deriv per bin
    edges       = np.linspace(X.min(), X.max(), n_bins + 1)
    midpoints   = (edges[:-1] + edges[1:]) / 2
    # assign each finite‐diff derivative to a bin by the midpoint of its grid‐interval
    deriv_mid_x = (grid[:-1] + grid[1:]) / 2
    bins_ix     = np.digitize(deriv_mid_x, edges) - 1  # 0..n_bins-1
    avg_deriv   = [
        deriv[bins_ix == i].mean() if np.any(bins_ix == i) else np.nan
        for i in range(n_bins)
    ]

    # 4) build interval DataFrame
    intervals = pd.DataFrame({
        'interval_start': edges[:-1],
        'interval_end'  : edges[1:],
        'avg_derivative': avg_deriv,
        'midpoint_x'    : midpoints
    })

    # 5) plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(X, y, alpha=0.3, s=20, label='data')
    ax.plot(grid, yhat, 'r-', linewidth=2, label='GAM fit')
    # vertical bin lines
    for e in edges:
        ax.axvline(e, color='gray', linestyle='--', linewidth=0.8)
    # annotate avg derivative
    y_top = ax.get_ylim()[1]
    for mp, ad in zip(midpoints, avg_deriv):
        ax.text(mp, y_top*0.95, f"{ad:.2f}", 
                ha='center', va='top', fontsize=8, rotation=90)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"GAM: {y_col} ~ s({x_col})")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return intervals

#%% Nepal plots
nepal_pairs = [
    ("Q0__hope_total__continuous",        "Q0__AGR_PROD__continuous"),
    ("Q0__self_control_score__continuous", "Q0__AGR_PROD__continuous"),
    ("Q0__average_of_farming_practices__ordinal","Q0__AGR_PROD__continuous"),
    ("Q0__Positive_Negative_Score__continuous","Q0__AGR_PROD__continuous"),
]

for Xcol, Ycol in nepal_pairs:
    print("\nNepal:", Xcol, "→", Ycol)
    ints = plot_gam_with_bins(nepal_df, Xcol, Ycol,
                              n_bins=10, n_grid=200, spline_k=25)
    print(ints)  # shows your 10 intervals + avg derivatives

#%% Senegal: build Q0_Avg_practices
from ast import literal_eval

def parse_list_cell(cell):
    if pd.isna(cell): return []
    if isinstance(cell, list): return cell
    try:
        return list(literal_eval(cell))
    except:
        return [s.strip() for s in str(cell).replace(";",",").split(",") if s.strip()]

all_pracs = set()
senegal_df["Q74__Practice__nominal"].dropna().map(parse_list_cell).apply(all_pracs.update)
max_pracs = len(all_pracs) or 1

senegal_df["Q0_Avg_practices"] = (
    senegal_df["Q74__Practice__nominal"]
              .map(parse_list_cell)
              .map(lambda L: len(L)/max_pracs)
)

senegal_pairs = [
    ("Q0__Hope_total__continuous", "Q0__AGR_PROD__continuous"),
    ("Q0__Average_CS__continuous", "Q0__AGR_PROD__continuous"),
    ("Q0_Avg_practices",           "Q0__AGR_PROD__continuous"),
]

for Xcol, Ycol in senegal_pairs:
    print("\nSenegal:", Xcol, "→", Ycol)
    ints = plot_gam_with_bins(senegal_df, Xcol, Ycol,
                              n_bins=10, n_grid=200, spline_k=25)
    print(ints)

# %%
