import numpy as np
import pandas as pd
import pingouin as pg
from factor_analyzer import FactorAnalyzer
from collections import defaultdict

from helpers import parse_feature_metadata

if __name__=="__main__":
    
    # load datasets
    nepal_df = pd.read_excel("nepal_dataframe_FA.xlsx", index_col=0)
    senegal_df = pd.read_excel("senegal_dataframe_FA.xlsx", index_col=0)

    for df in (nepal_df, senegal_df):
        
        # TODO: change any code above next todo that might cause issues
        y_prod = df["Q0__AGR_PROD__continuous"]
        y_sus = df["Q0__sustainable_livelihood_score__continuous"]
        df.drop(columns=["Q0__AGR_PROD__continuous", "Q0__sustainable_livelihood_score__continuous"], axis=1, inplace=True)

        # drop unnecessary
        # split data by variable type
        col_type_map = defaultdict(list)

        for col in df.columns:
            t = parse_feature_metadata(col)["type"]
            col_type_map[t].append(col)

        # continuous already scaled from EDA, no need to scale here
        df_num = df[col_type_map["continuous"] + col_type_map["ordinal"] + col_type_map["binary"]].copy()

        # build mixed correlation matrix
        R = np.eye(df_num.shape[1])
        idx = {col: i for i, col in enumerate(df_num.columns)}

        # pearson blocks
        R[np.ix_([idx[c] for c in col_type_map["continuous"]+col_type_map["binary"]],
         [idx[c] for c in col_type_map["continuous"]+col_type_map["binary"]])] = np.corrcoef(
            df_num[col_type_map["continuous"]+col_type_map["binary"]], rowvar=False)

        # polychoric block for ordinals TODO: change this and use semopy to calculate polychoric correlation because pingouin does not have this 
        if col_type_map["ordinal"]:
            R_ord = pg.polychoric(df_num[col_type_map["ordinal"]])[0]
            R[np.ix_([idx[c] for c in col_type_map["ordinal"]],
                    [idx[c] for c in col_type_map["ordinal"]])] = R_ord
            # Pearson cross-blocks (cont|bin vs ordinal) are fine

        # TODO: tetrachoric block for binary features

        # TODO: parallel analysis to determine how many factors to retain

        # TODO: extract + rotate + score