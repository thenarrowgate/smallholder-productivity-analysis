# ────────────────────────────────────────────────────────────────────────────────
# Load required packages
library(readxl)     # read_excel()
library(dplyr)      # data wrangling
library(stringr)    # str_split()
library(Gifi)       # princals()
library(psych)      # fa.parallel(), fa(), factor.congruence()
library(EFAtools)   # VSS()
library(boot)       # bootstrap()
library(mgcv)       # gam()
library(lavaan)     # sem()
library(latentcor)  # latentcor()
# ────────────────────────────────────────────────────────────────────────────────

set.seed(2025)

# 1. Load data
df <- read_excel("nepal_dataframe_FA.xlsx")

# Extract outcome and drop ancillary score
y_prod <- df$Q0__AGR_PROD__continuous
df      <- df %>% 
  select(-Q0__AGR_PROD__continuous,
         -Q0__sustainable_livelihood_score__continuous)

# 2. Split by type
types <- str_split(names(df), "__", simplify = TRUE)[,3]
types[types == "binary_nominal"] <- "nominal"

df_cont <- df[, types == "continuous", drop = FALSE]
df_ord  <- df[, types == "ordinal",    drop = FALSE]
df_bin  <- df[, types == "binary",     drop = FALSE]
df_nom  <- df[, types == "nominal",    drop = FALSE]

# 3. Optimal-scale high-cardinal nominals via princals()
nom_levels <- sapply(df_nom, function(x) length(unique(x)))
high_nom   <- names(nom_levels[nom_levels > 6])

if (length(high_nom) > 0) {
  df_nom[high_nom] <- lapply(df_nom[high_nom], factor)
  m <- length(high_nom)
  pc_out <- princals(
    data    = as.data.frame(df_nom[high_nom]),
    ndim    = m,
    levels  = rep("nominal", m),
    verbose = FALSE
  )
  df_quant <- as.data.frame(pc_out$objectscores)
  names(df_quant) <- paste0(high_nom, "_quant")
  df_cont <- bind_cols(df_cont, df_quant)
  df      <- df %>% select(-all_of(high_nom))
}

# 4. Standardize any quant scores
quant_cols <- grep("_quant$", names(df_cont), value = TRUE)
if (length(quant_cols)) {
  df_cont[quant_cols] <- lapply(df_cont[quant_cols], scale)
}

# 5. Convert ordinal and binary to integer codes
df_ord_num <- df_ord %>% mutate(across(everything(), as.integer))
df_bin_num <- df_bin %>% 
  mutate(across(everything(), ~ as.integer(as.character(.))))

# 6. Build analysis data frame (all numeric)
df_mix2_clean <- bind_cols(df_cont, df_ord_num, df_bin_num) %>%
  select(where(~ !any(is.na(.)))) %>%
  as.data.frame()

# 7. Estimate latent-Gaussian correlation matrix
lc_out <- latentcor(
  X         = df_mix2_clean,
  types     = NULL,        # auto-detect types
  method    = "approx",    # fast inversion
  use.nearPD= TRUE         # enforce PD
)
R_mixed <- lc_out$R
stopifnot(!any(is.na(R_mixed)))

# 8. Eigenvalue check (before EFA)
ev <- eigen(R_mixed)$values
plot(ev, main="Eigenvalues (latentcor)", xlab="Index", ylab="Eigenvalue")

# 9. Parallel analysis & MAP to pick k
pa_out   <- fa.parallel(R_mixed, n.obs = n,
                        fm = "minres", fa = "fa",
                        n.iter = 500, quant = .95,
                        cor = "cor", use = "pairwise",
                        plot = FALSE)
k_PA     <- pa_out$nfact
vss_out  <- VSS(R_mixed, n = ncol(R_mixed),
                fm = "minres", n.obs = n, plot = FALSE)
k_MAP    <- which.min(vss_out$map)
k        <- k_MAP  # choose as desired

# 10. Initial MINRES + oblimin
efa_init <- fa(R_mixed, nfactors = k,
               fm = "minres", rotate = "oblimin", n.obs = n)

# 11. Prune low‐communality items
h2      <- efa_init$communality
keep    <- names(h2)[h2 >= .30]
R_prune <- R_mixed[keep, keep]
efa0    <- fa(R_prune, nfactors = k,
              fm = "minres", rotate = "oblimin", n.obs = n)
Lambda0 <- efa0$loadings
Psi0    <- efa0$uniquenesses

# 12. Bootstrap for Tucker’s φ & Hancock’s H using latentcor
B         <- 1000
phis      <- matrix(NA, B, k)
Hs        <- matrix(NA, B, k)
completed <- 0
set.seed(2025)

all_vars <- names(df_mix2_clean)
stopifnot(all(keep %in% all_vars))

while (completed < B) {
  # 12.1 Resample rows
  samp_idx <- sample(nrow(df_mix2_clean), replace = TRUE)
  df_samp  <- df_mix2_clean[samp_idx, , drop = FALSE]
  
  # — New check: skip if **any** column is constant in this sample —
  zero_vars_all <- names(df_samp)[
    sapply(df_samp, function(x) var(x, na.rm = TRUE) == 0)
  ]
  if (length(zero_vars_all) > 0) {
    # (Optional) print which vars are constant for debugging
    # cat("Skipping sample; constant vars:", paste(zero_vars_all, collapse = ", "), "\n")
    next
  }
  
  # 12.2 Compute full latent‐Gaussian correlation on the sample
  lc_b <- latentcor(
    X         = df_samp,
    types     = NULL,
    method    = "approx",
    use.nearPD= TRUE
  )
  R_full <- lc_b$R
  
  # 12.3 Subset to pruned variables only
  Rb <- R_full[keep, keep]
  
  # 12.4 Fit MINRES + oblimin
  fa_b <- tryCatch(
    fa(Rb,
       nfactors = k,
       fm       = "minres",
       rotate   = "oblimin",
       n.obs    = nrow(df_samp)),
    error = function(e) NULL
  )
  if (is.null(fa_b)) next
  
  # 12.5 Record congruence & H-index
  completed       <- completed + 1
  Lb              <- fa_b$loadings
  psib            <- fa_b$uniquenesses
  phis[completed, ] <- diag(factor.congruence(Lambda0, Lb))
  Hs[completed, ]   <- vapply(seq_len(k), function(j) {
    sum(Lb[, j])^2 / (sum(Lb[, j])^2 + sum(psib))
  }, numeric(1))
}

# 13. Report stability
cat("Mean Tucker’s φ: ", colMeans(phis), "\n")
cat("Mean Hancock’s H:", colMeans(Hs),   "\n")

