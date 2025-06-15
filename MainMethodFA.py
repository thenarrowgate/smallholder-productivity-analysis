
#%%
# MainMethodFA.py
import monkey_patch_mvnun  # ← must come first!
# now all following imports will see mvn.mvnun available again
from semopy.polycorr import polychoric_corr, polyserial_corr

import numpy as np
import pandas as pd
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer import FactorAnalyzer, calculate_kmo
from collections import defaultdict
from IPython.display import display

import matplotlib.pyplot as plt


from helpers import parse_feature_metadata
#%%

NUM_FACTORS = 5

#%%
# Parallel Analysis helper (Horn, 1965)
def parallel_analysis(data, n_iter=100, percentile=95):
    n, m = data.shape
    fa_ev = FactorAnalyzer(n_factors=m,
                           rotation=None,
                           method='principal',
                           use_smc=True,
                           is_corr_matrix=False)
    fa_ev.fit(data)
    real_ev, _ = fa_ev.get_eigenvalues()

    rand_evs = np.zeros((n_iter, m))
    for i in range(n_iter):
        rand = np.random.normal(size=(n, m))
        fa_ev.fit(rand)
        rand_evs[i], _ = fa_ev.get_eigenvalues()
    rand_pct_ev = np.percentile(rand_evs, percentile, axis=0)

    n_factors = np.sum(real_ev > rand_pct_ev)

    plt.figure()
    plt.plot(range(1, m+1), real_ev, 'o-', label='Real Data')
    plt.plot(range(1, m+1), rand_pct_ev, 'o--', label=f'Random {percentile}th pct')
    plt.axhline(1, color='grey', linestyle='--', label='Eigenvalue = 1')
    plt.xlabel('Component')
    plt.ylabel('Eigenvalue')
    plt.title('Parallel Analysis & Scree Plot')
    plt.legend()
    plt.show()

    return n_factors, real_ev

def build_mixed_corr(df_num, cont_cols, bin_cols, ord_cols):
    """
    Construct mixed-type correlation matrix:
      - cont–cont: Pearson
      - cont–ord or cont–bin: polyserial
      - ord–ord, ord–bin, bin–bin: polychoric
    """
    cols = list(df_num.columns)
    Rm = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)

    for i, c1 in enumerate(cols):
        for j in range(i+1, len(cols)):
            c2 = cols[j]

            if c1 in cont_cols and c2 in cont_cols:
                corr = df_num[c1].corr(df_num[c2], method='pearson')
            elif (c1 in cont_cols and c2 in ord_cols) or (c2 in cont_cols and c1 in ord_cols):
                # continuous–ordinal
                x = df_num[c1].values if c1 in cont_cols else df_num[c2].values
                y = df_num[c2].values if c2 in ord_cols else df_num[c1].values
                corr = polyserial_corr(x, y)
            elif (c1 in cont_cols and c2 in bin_cols) or (c2 in cont_cols and c1 in bin_cols):
                # continuous–binary (biserial ≈ polyserial)
                x = df_num[c1].values if c1 in cont_cols else df_num[c2].values
                y = df_num[c2].values if c2 in bin_cols else df_num[c1].values
                corr = polyserial_corr(x, y)
            else:
                # ord–ord, ord–bin, bin–bin: tetra/polychoric
                corr = polychoric_corr(df_num[c1].values, df_num[c2].values)
            Rm.iat[i, j] = corr
            Rm.iat[j, i] = corr

    return Rm

def spearman_corr_EFA():

    # load datasets
    nepal_df = pd.read_excel("nepal_dataframe_FA.xlsx", index_col=0)
    senegal_df = pd.read_excel("senegal_dataframe_FA.xlsx", index_col=0)

    for df in (nepal_df, senegal_df):
        
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
        df_continuous = df[col_type_map["continuous"]].copy()
        df_binary = df[col_type_map["binary"]].copy()
        df_ordinal = df[col_type_map["ordinal"]].copy()
        df_nominal = df[col_type_map["ordinal"]].copy()

        # Construct Spearman correlation matrix for EFA
        # Using pandas.DataFrame.corr(method='spearman') to compute rank-based correlations
        R = df_num.corr(method="spearman")

        # Perform Bartlett's test of sphericity to assess factorability
        # H0: correlation matrix is identity (no relationships among variables)
        from scipy.stats import chi2
        n, p = df.shape
        det_reg = np.linalg.det(R)
        stat = -np.log(det_reg) * (n - 1 - (2*p + 5)/6)
        pval = chi2.sf(stat, p*(p-1)/2)
        print(f"Bartlett's Test Chi-square: {stat}, p-value: {pval}")

        kmo_per_var, kmo_model = calculate_kmo(df_num)   
        print(f"KMO Total = {kmo_model:.3f}")

        kmo_df = pd.DataFrame({
            "Variable": df_num.columns,
            "KMO": kmo_per_var
        })

        # 3. Sort and display
        kmo_df = kmo_df.set_index("Variable").sort_values("KMO", ascending=False)
        display(kmo_df)

        parallel_analysis(df_num.values,
                            n_iter=100,
                            percentile=95)
        
        fa = FactorAnalyzer(n_factors=NUM_FACTORS,
                            method='minres', # principal axis
                            rotation='oblimin',
                            use_smc=True,
                            is_corr_matrix=True)
        fa.fit(R)

        loadings      = fa.loadings_
        communalities = fa.get_communalities()

        factors = [f"F{i+1}" for i in range(NUM_FACTORS)]
        load_df = pd.DataFrame(loadings,
                               index=df_num.columns,
                               columns=factors)
        comm_df = pd.DataFrame(communalities,
                               index=df_num.columns,
                               columns=["Communality"])
        result  = pd.concat([load_df, comm_df], axis=1)

        # 1) define a formatter for numeric precision
        fmt = "{:.3f}".format

        # 2) define a highlighter for loadings > 0.4
        def highlight_pos(val):
            return "color: red;" if isinstance(val, (int, float)) and abs(val) > 0.4 else ""

        # 3) apply styling
        styled = (
            result.style
                .format(fmt)                            # format all numbers to 3 decimals
                .applymap(highlight_pos, subset=factors)  # only on the factor columns
                .set_properties(**{
                    'border': '1px solid lightgrey',
                    'padding': '4px'
                })
                .set_table_styles([{
                    'selector': 'th',
                    'props': [('background-color', "#181818"),
                                ('border', '1px solid lightgrey'),
                                ('padding', '4px')]

                }])
        )

        # 4) display it
        print("\nFactor Loadings & Communalities:")
        display(styled)

        # TODO

def mixed_corr_EFA():

    # load datasets
    nepal_df = pd.read_excel("nepal_dataframe_FA.xlsx", index_col=0)
    senegal_df = pd.read_excel("senegal_dataframe_FA.xlsx", index_col=0)

    for df in (nepal_df, senegal_df):

        # separate outcomes
        y_prod = df.pop("Q0__AGR_PROD__continuous")
        y_sus  = df.pop("Q0__sustainable_livelihood_score__continuous")

        # split by metadata types
        col_type_map = defaultdict(list)
        for col in df.columns:
            t = parse_feature_metadata(col)["type"]
            col_type_map[t].append(col)

        cont_cols = col_type_map["continuous"]
        bin_cols  = col_type_map["binary"]
        ord_cols  = col_type_map["ordinal"]

        df_num = df[cont_cols + bin_cols + ord_cols].copy()

        # --- TODO: construct mixed correlation matrix ---
        Rm = build_mixed_corr(df_num, cont_cols, bin_cols, ord_cols)

        # Bartlett's test on mixed Rm
        chi2_stat, bart_p = calculate_bartlett_sphericity(Rm)
        print(f"\nMixed Bartlett's Test  χ² = {chi2_stat:.2f}, p = {bart_p:.3f}")

        # KMO on raw data (uses underlying pearson in factor_analyzer)
        kmo_per, kmo_tot = calculate_kmo(df_num)
        print(f"Mixed KMO Total = {kmo_tot:.3f}")
        display(pd.DataFrame({"KMO": kmo_per}, index=df_num.columns).sort_values("KMO", ascending=False))

        # Parallel Analysis
        n_factors, real_ev = parallel_analysis(df_num.values, n_iter=100, percentile=95)

        # Factor extraction on mixed Rm
        fa = FactorAnalyzer(n_factors=n_factors,
                            method='wls',
                            rotation='oblimin',
                            use_smc=True,
                            is_corr_matrix=True)
        fa.fit(Rm.values)

        # Display loadings & communalities (styled)
        loadings      = fa.loadings_
        communalities = fa.get_communalities()
        factors = [f"F{i+1}" for i in range(n_factors)]

        result = pd.DataFrame(loadings, index=df_num.columns, columns=factors)
        result["Communality"] = communalities

        # formatter & highlighter
        fmt = "{:.3f}".format
        def highlight_pos(val):
            return "color: red;" if isinstance(val, (int, float)) and abs(val) > 0.4 else ""

        styled = (result.style
                      .format(fmt)
                      .applymap(highlight_pos, subset=factors)
                      .set_properties(border='1px solid lightgrey', padding='4px')
                      .set_table_styles([{
                          'selector': 'th',
                          'props': [('background-color', "#181818"),
                                    ('border', '1px solid lightgrey'),
                                    ('padding', '4px')]
                      }]))
        print("\nMixed Data Factor Loadings & Communalities:")
        display(styled)

        # TODO: print the 

#%%
if __name__=="__main__":
    spearman_corr_EFA()
    

# %%
