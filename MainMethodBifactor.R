# ─── 0. Load packages ───────────────────────────────────────────────────────────
library(readxl)    # read_excel()
library(dplyr)     # data wrangling
library(stringr)   # str_split()
library(psych)     # fa.parallel(), factor.congruence()
library(bifactor)  # exploratory bi‐factor EFA (Jiménez et al., 2023)
set.seed(2025)

# ─── 1. Read & preprocess ───────────────────────────────────────────────────────
df <- read_excel("E:/Atuda/67814-Data-Science-Final-Project/Code/nepal_dataframe_FA.xlsx")

# Keep only continuous/ordinal/binary
types  <- str_split(names(df), "__", simplify = TRUE)[,3]
types[types=="binary_nominal"] <- "nominal"
df_num <- bind_cols(
  df[, types=="continuous", drop=FALSE],
  df[, types=="ordinal",    drop=FALSE],
  df[, types=="binary",     drop=FALSE]
)

# Rank‐transform for Spearman EFA
df_rank <- as.data.frame(
  lapply(df_num, function(col) rank(col, ties.method="average"))
)

# ─── 2. Parallel analysis → total factors ────────────────────────────────────────
R_full   <- cor(df_rank, method="spearman", use="pairwise.complete.obs")
pa       <- fa.parallel(R_full, n.obs   = nrow(df_rank),
                        fm      = "minres", fa      = "fa",
                        n.iter  = 500,    quant   = .95,
                        cor     = "spearman", plot = FALSE)
K_total  <- pa$nfact   # total dimensions = 1 general + (K_total-1) groups

# ─── 3. Fit Exploratory Bi-Factor EFA via bifactor() with PhiTarget ───────────
q <- 1 + (K_total - 1)  # total # of factors

ebfa_mod <- bifactor(
  X            = R_full,         # pass your Spearman correlation matrix
  n_generals   = 1,
  n_groups     = K_total - 1,
  method       = "GSLiD",
  cor          = "pearson",      # since X is already a correlation matrix
  estimator    = "uls",
  projection   = "oblq",
  missing      = "none",
  nobs         = nrow(df_rank),
  PhiTarget    = diag(q),        # ← here’s the identity target for Phi
  random_starts= 10,
  cores        = 1,
  verbose      = FALSE
)

# Extract loadings & uniquenesses
B0   <- ebfa_mod$bifactor$lambda        # p × (1 + (K_total-1))
uni0 <- ebfa_mod$bifactor$uniquenesses  # length‐p
G    <- ncol(B0)             # total factors

# ─── 4. Bootstrap φ & H‐index ───────────────────────────────────────────────────
B      <- 1000
phis   <- matrix(NA, B, G)
Hs     <- matrix(NA, B, G)
done   <- 0

while (done < B) {
  # 1. Sample rows
  idx <- sample(nrow(df_rank), replace = TRUE)
  Rb  <- tryCatch(
    cor(df_rank[idx, ], method = "spearman", use = "pairwise.complete.obs"),
    error = function(e) NULL
  )
  if (is.null(Rb) || any(is.na(Rb))) next
  
  # 2. Refit EBFA on Rb
  fit <- tryCatch(
    bifactor(
      X           = Rb,
      n_generals  = 1,
      n_groups    = K_total - 1,
      method      = "GSLiD",
      cor         = "pearson",
      estimator   = "uls",
      projection  = "oblq",
      missing     = "none",
      nobs        = nrow(df_rank),
      PhiTarget   = diag(1 + (K_total-1)),
      random_starts= 5,
      cores       = 1,
      verbose     = FALSE
    ),
    error = function(e) NULL
  )
  if (is.null(fit)) next
  done <- done + 1
  
  # 3. Extract replicate loadings & uniquenesses
  Bb   <- fit$bifactor$lambda
  psi_b<- fit$bifactor$uniquenesses
  psi_b[psi_b < 0] <- 0
  
  # 4a. Tucker’s φ
  phis[done, ] <- diag(psych::factor.congruence(B0, Bb))
  
  # 4b. Hancock’s H‐index
  Hs[done, ]   <- sapply(seq_len(G), function(j) {
    lam2 <- Bb[, j]^2
    sum(lam2) / (sum(lam2) + sum(psi_b))
  })
}


# ─── 5. Summarize & print ───────────────────────────────────────────────────────
phi_means <- colMeans(phis, na.rm=TRUE)
H_means   <- colMeans(Hs,   na.rm=TRUE)

cat("Bi-factor Tucker’s φ (general + group):\n");      print(phi_means)
cat("\nBi-factor Hancock’s H-index (general + group):\n"); print(H_means)
