#%%
import pandas as pd
import numpy as np
from scipy.stats import shapiro, levene
from itertools import combinations, product
import warnings
from collections import defaultdict

from helpers import parse_feature_metadata
#%%
def check_normality(x, alpha=0.05):
    """Return True if Shapiro–Wilk p-value > alpha."""
    stat, p = shapiro(x)
    return p > alpha, p

def check_variance_equality(groups, alpha=0.05):
    """Return True if Levene's test p-value > alpha."""
    stat, p = levene(*groups)
    return p > alpha, p

def check_category_freq(s, min_count=10):
    """Return True if all categories have at least min_count observations."""
    freqs = s.value_counts(dropna=False)
    return bool((freqs >= min_count).all()), freqs

def check_contingency_table(a, b, min_cell=5, continuity=0.5):
    """Check 2D crosstab cell counts; return corrected table if zeros."""
    ct = pd.crosstab(a, b)
    if (ct.values < min_cell).any():
        warnings.warn("Some cells < %d; applying continuity correction." % min_cell)
        ct = ct + continuity
    return ct

def perform_assumption_checks(df_cont, df_bin, df_ord):
    results = []

    # 1. Continuous–Continuous
    for x, y in combinations(df_cont.columns, 2):
        ok_x, p_x = check_normality(df_cont[x])
        ok_y, p_y = check_normality(df_cont[y])
        results.append({
            'type': 'cont–cont', 'vars': (x, y),
            'normal_x': p_x, 'normal_y': p_y
        })

    # 2. Continuous–Binary
    for x in df_cont.columns:
        for b in df_bin.columns:
            overall_ok, p_over = check_normality(df_cont[x])
            group_ok, p_group = True, {}
            # per-group normality
            for lvl, grp in df_cont.groupby(df_bin[b]):
                ok, p = check_normality(grp[x])
                p_group[lvl] = p
                group_ok &= ok
            # varaince equal between groups
            var_ok, p_var = check_variance_equality([
                df_cont.loc[df_bin[b]==lvl, x] for lvl in df_bin[b].unique()
            ])

            results.append({
                'type': 'cont–bin', 'vars': (x, b),
                'normal_overall_p': p_over,
                'normal_by_group_p': p_group,
                'levene_p': p_var
            })

    # 3. Continuous–Ordinal
    for x in df_cont.columns:
        for o in df_ord.columns:
            cont_ok, p_cont = check_normality(df_cont[x])
            freq_ok, freqs = check_category_freq(df_ord[o])
            results.append({
                'type': 'cont–ord', 'vars': (x, o),
                'normal_cont_p': p_cont,
                'category_freqs': freqs.to_dict()
            })

    # 4. Ordinal–Ordinal & Ordinal–Binary
    for df1, df2, corr_type in [
        (df_ord, df_ord, 'ord–ord'),
        (df_ord, df_bin, 'ord–bin'),
    ]:
        for a, b in product(df1.columns, df2.columns):
            freq1_ok, f1 = check_category_freq(df1[a])
            freq2_ok, f2 = check_category_freq(df2[b])
            ct = pd.crosstab(df1[a], df2[b])
            if (ct.values == 0).any():
                warnings.warn(f"Zero cell in {a}-{b}; will need correction.")
            results.append({
                'type': corr_type, 'vars': (a, b),
                'freqs_1': f1.to_dict(), 'freqs_2': f2.to_dict(),
                'contingency': ct.to_dict()
            })

    # 5. Binary–Binary
    for a, b in combinations(df_bin.columns, 2):
        ct = pd.crosstab(df_bin[a], df_bin[b])
        if (ct.values < 5).any():
            warnings.warn(f"Cell count <5 in {a}-{b}; applying continuity corr.")
            ct = ct + 0.5
        results.append({
            'type': 'bin–bin', 'vars': (a, b),
            'contingency_corrected': ct.to_dict()
        })

    return pd.DataFrame(results)

#%%
nepal_df = pd.read_excel("nepal_dataframe_FA.xlsx", index_col=0)
senegal_df = pd.read_excel("senegal_dataframe_FA.xlsx", index_col=0)

for df, country in zip((nepal_df, senegal_df), ("NPL", "SGL")):
    
    print(country)
    
    y_prod = df["Q0__AGR_PROD__continuous"]
    y_sus = df["Q0__sustainable_livelihood_score__continuous"]
    df.drop(columns=["Q0__AGR_PROD__continuous", "Q0__sustainable_livelihood_score__continuous"], axis=1, inplace=True)

    # drop unnecessary
    # split data by variable type
    col_type_map = defaultdict(list)

    for col in df.columns:
        t = parse_feature_metadata(col)["type"]
        col_type_map[t].append(col)

    df_num = df[col_type_map["continuous"] + col_type_map["binary"] + col_type_map["ordinal"]]
    df_continuous = df[col_type_map["continuous"]]
    df_binary = df[col_type_map["binary"]]
    df_ordinal = df[col_type_map["ordinal"]]
    df_nom = df[col_type_map["nominal"]]

    #checks_df = perform_assumption_checks(df_continuous, df_binary, df_ordinal)
    #print(checks_df)

    print(df_nom.shape)
    print(df_nom.columns)

# %%
#checks_df[(checks_df["type"] == 'cont–bin') & (np.array([v > 0.05 for v in checks_df["normal_by_group_p"].values()]).any()]
#%%
checks_df[checks_df["type"] == 'cont–bin']['normal_by_group_p'].apply(
    lambda d: max(d.values()) > 0.05 if isinstance(d, dict) and d else None
).value_counts()
#%%
checks_df[(checks_df["type"] == 'cont–bin') & (checks_df["levene_p"] > 0.05)]
#%%
checks_df[checks_df["type"] == 'cont–ord']['category_freqs'].apply(
    lambda d: min(d.values()) < 10 if isinstance(d, dict) and d else None
).value_counts()
#%%
checks_df[(checks_df["type"] == 'ord–ord') | (checks_df["type"] == 'ord–ord')]
#%%
checks_df[checks_df["type"] == 'bin–bin']
#%%

df = pd.read_excel("nepal_dataframe_FA.xlsx", index_col=0)


col_type_map = defaultdict(list)

for col in df.columns:
    t = parse_feature_metadata(col)["type"]
    col_type_map[t].append(col)

df_nom = df[col_type_map["nominal"]]
#%%
df_nom.isna().sum()
# %%
