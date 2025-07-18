# load requires packages
library(readxl)    # Excel I/O
library(dplyr)     # Data wrangling
library(stringr)   # Regex utilities
library(EFAtools)  # EFA retention criteria, PARALLEL, VSS
library(psych)     # fa.parallel, fa(), factor.congruence()
library(boot)      # bootstrap()
library(mgcv)      # gam()
library(lavaan)    # sem(), fitMeasures()
library(EFA.dimensions)
set.seed(2025)

# load data
setwd("E:/Atuda/67814-Data-Science-Final-Project/Code")
df <- read_excel("nepal_dataframe_FA.xlsx")

process_df <- function(df) {
  # Extract outcomes
  y_prod <- df$Q0__AGR_PROD__continuous
  df      <- df %>% select(-Q0__AGR_PROD__continuous,
                           -Q0__sustainable_livelihood_score__continuous)
  # Parse types via regex on column names
  types <- str_split(names(df), "__", simplify = TRUE)[,3]
  # Map binary_nominal → nominal
  types[types=="binary_nominal"] <- "nominal"
  # Split columns
  df_cont   <- df[, types=="continuous"]
  df_ord    <- df[, types=="ordinal"]
  df_bin    <- df[, types=="binary"]
  df_nom    <- df[, types=="nominal"]
  df_num    <- bind_cols(df_cont, df_ord, df_bin)
  list(df_num = df_num, df_nom = df_nom, y_prod = y_prod)
}

nepal  <- process_df(df)

n <- nrow(df)
df_num = nepal$df_num


# Compute Spearman R
R <- cor(df_num, method = "spearman", use = "pairwise.complete.obs")

# Parallel analysis (minres factors) with 500 iterations, 95% quantile
fa.parallel(R,
            n.obs    = nrow(df_num),
            fm       = "minres",
            fa       = "fa",
            n.iter   = 500,
            quant    = .95,
            cor      = "spearman",
            plot     = FALSE) -> pa_out

k_PA <- pa_out$nfact    # suggested upper bound :contentReference[oaicite:6]{index=6}

# Compute VSS with minimum-residual EFA (as in your pipeline)
vss_res <- VSS(R,
               n       = ncol(R),
               fm      = "minres",
               n.obs   = nrow(df_num),
               plot    = FALSE)

# Extract the MAP values and choose the minimum
k_MAP <- which.min(vss_res$map)

k <- k_MAP

# 0. Start from your original numeric data
#    (not the ranked version yet)
df_num_clean <- df_num[, colSums(is.na(df_num)) == 0]

# 1. Rank‐transform for Spearman
df_rank <- as.data.frame(
  lapply(df_num_clean, rank, ties.method = "average")
)

# 1. Fit the original MINRES+oblimin on the ranks
efa_to_prune <- psych::fa(
  df_rank,
  nfactors = k,
  fm       = "minres",
  rotate   = "oblimin",
  use      = "pairwise.complete.obs"
)

# Identify items with communality >= 0.40
h2    <- efa_to_prune$communality
# Keep those items, plus parcels created above
keep  <- names(h2)[abs(h2) >= 0.30]

df_rank <- df_rank[, keep]

efa_0 <- psych::fa(
  df_rank,
  nfactors = k,
  fm       = "minres",
  rotate   = "oblimin",
  use      = "pairwise.complete.obs"
)
Lambda0 <- efa_0$loadings
Psi0    <- efa_0$uniquenesses

# 2. Bootstrap but skip any failures
B         <- 1000
phis      <- matrix(NA, nrow = B, ncol = k)
Hs        <- matrix(NA, nrow = B, ncol = k)
completed <- 0
set.seed(2025)

while (completed < B) {
  # draw a bootstrap sample of rows
  samp <- df_rank[sample(nrow(df_rank), replace = TRUE), , drop = FALSE]
  
  # safe EFA call
  fa_b <- tryCatch(
    psych::fa(
      samp,
      nfactors = k,
      fm       = "minres",
      rotate   = "oblimin",
      use      = "pairwise.complete.obs"
    ),
    error = function(e) NULL
  )
  if (is.null(fa_b)) next   # skip this replicate & retry
  
  # successful replicate
  completed <- completed + 1
  
  Lb   <- fa_b$loadings
  psib <- fa_b$uniquenesses
  
  # Tucker's phi
  phis[completed, ] <- diag(
    psych::factor.congruence(Lambda0, Lb)
  )
  
  # Hancock's H-index
  Hs[completed, ] <- vapply(
    seq_len(k),
    function(j) {
      sum(Lb[, j])^2 /
        ( sum(Lb[, j])^2 + sum(psib) )
    },
    numeric(1)
  )
}

# 3. Aggregate
phi_means <- colMeans(phis)
H_means   <- colMeans(Hs)

print(phi_means)  # expect ≥ 0.90
print(H_means)    # expect ≥ 0.80

