import pandas as pd

NEPAL_PATH_NADAV = r"E:\Atuda\67814-Data-Science-Final-Project\Code\smallholder-productivity-analysis\data\raw\nepal_new_feature_names.xlsx"
SENEGAL_PATH_NADAV = r"E:\Atuda\67814-Data-Science-Final-Project\Code\smallholder-productivity-analysis\data\raw\senegal_new_feature_names.xlsx"
NEPAL_PATH_NADAV_NEW = r"E:\Atuda\67814-Data-Science-Final-Project\Code\smallholder-productivity-analysis\data\raw\nepal_new_feature_names2.xlsx"
SENEGAL_PATH_NADAV_NEW = r"E:\Atuda\67814-Data-Science-Final-Project\Code\smallholder-productivity-analysis\data\raw\senegal_new_feature_names2.xlsx"

def main():
    

    nepal_df = pd.read_excel(NEPAL_PATH_NADAV)
    senegal_df = pd.read_excel(SENEGAL_PATH_NADAV)

    def sanitize(col):
        return col.replace('__', '_').replace('-', '__').replace('#', '0').replace(':', '').replace('.', '').replace(' ', '')

    nepal_df.columns = nepal_df.columns.map(sanitize)
    senegal_df.columns = senegal_df.columns.map(sanitize)

    nepal_df.to_excel(NEPAL_PATH_NADAV_NEW)
    senegal_df.to_excel(SENEGAL_PATH_NADAV_NEW)

if __name__ == "__main__":
    main()