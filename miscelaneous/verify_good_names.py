#%%
import pandas as pd
import re

df = pd.read_excel("senegal_dataframe_final.xlsx", index_col=0)
#%%
pattern = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

for col in df.columns:
    if len(col.split("__")) == 4 and col.split("__")[-2] != "binary_nominal":
        var_name = col  # you could also isolate just the "name" part if needed
        if not pattern.match(var_name):
            print(f"Non-standard column name: {col}")

# %%
for col in df.columns:
    if "Hope" in col:
        print(col)