import numpy as np
import pandas as pd

# 1) Fix random seed and number of samples
np.random.seed(42)
n = 500

# 2) Exogenous continuous variable A
A = np.random.uniform(-1, 1, n)

# 3) Noise terms for each equation
noise_B = np.random.normal(0, 0.1, n)
noise_C = np.random.normal(0, 0.1, n)
noise_E = np.random.normal(0, 0.1, n)
noise_F = np.random.normal(0, 0.1, n)
noise_G = np.random.normal(0, 0.1, n)
noise_H = np.random.normal(0, 1, n)
noise_J = np.random.normal(0, 1, n)

# 4) B depends linearly on A
B = 2.0 * A + noise_B

# 5) C depends nonlinearly on A (quadratic)
C = A**2 + noise_C

# 6) D is a binary threshold of B
B_median = np.median(B)
D = (B > B_median).astype(int)

# 7) E depends nonlinearly on (D, C)
E = np.where(D == 1, 1.5 * C + noise_E, -1.5 * C + noise_E)

# 8) F mixes a linear (B) and nonlinear (sin(C)) effect
F = 0.5 * B + np.sin(C) + noise_F

# 9) G is an ordinal variable derived from A + small noise
#    (we create three bins: values in (−∞,−0.5], (−0.5,0.5], (0.5,∞))
G_cont = A + noise_G
G = np.digitize(G_cont, bins=[-0.5, 0.5])  # yields {0, 1, 2}

# 10) H, I, J are independent noise
H = noise_H
I = np.random.binomial(1, 0.5, n)  # fair coin
J = noise_J

# 11) Put everything into a DataFrame with “__” in the names so parse_feature_metadata works
df_simulated = pd.DataFrame({
    "Q1__A__continuous": A,
    "Q1__B__continuous": B,
    "Q1__C__continuous": C,
    "Q1__D__binary":     D,
    "Q1__E__continuous": E,
    "Q1__F__continuous": F,
    "Q1__G__ordinal":    G,
    "Q1__H__continuous": H,
    "Q1__I__binary":     I,
    "Q1__J__continuous": J
})

from sklearn.preprocessing import StandardScaler

# pick up continuous _and_ ordinal
scale_cols = [
    col for col in df_simulated.columns 
    if "__continuous" in col or "__ordinal" in col
]

scaler = StandardScaler()
df_simulated[scale_cols] = scaler.fit_transform(df_simulated[scale_cols])

# binary columns remain untouched
df_simulated.to_excel("dummy.xlsx", index=False)

