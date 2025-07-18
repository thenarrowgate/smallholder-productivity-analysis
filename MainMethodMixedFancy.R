# Load required packages
library(dplyr)       # data wrangling
library(EFAtools)    # VSS()
library(boot)        # bootstrap()
library(Gifi)        # princals()
library(lavaan)      # sem()
library(mgcv)        # gam()
library(polycor)     # hetcor()
library(psych)       # mixedCor(), fa.*, factor.congruence()
library(readxl)      # read_excel()
library(stringr)     # str_split()
library(doSNOW)      # parallel backend for foreach
library(foreach)     # foreach looping

set.seed(2025)

# 1. Load data
LOCAL_DIR <- "E:/Atuda/67814-Data-Science-Final-Project/Code"
setwd(LOCAL_DIR)
(df <- read_excel("nepal_dataframe_FA.xlsx"))

# 2. Extract outcome and remove from df
y_prod <- df$Q0__AGR_PROD__continuous
df <- df %>% select(-Q0__AGR_PROD__continuous,
                    -Q0__sustainable_livelihood_score__continuous)

# 3. Split into continuous / ordinal / binary / nominal
(types  <- str_split(names(df), "__", simplify = TRUE)[,3])
types[types == "binary_nominal"] <- "nominal"

df_cont <- df[, types == "continuous", drop = FALSE]
df_ord  <- df[, types == "ordinal",    drop = FALSE]
df_bin  <- df[, types == "binary",     drop = FALSE]
df_nom  <- df[, types == "nominal",    drop = FALSE]

# 4. Factor df_ord and df_bin as ordered
(df_ord_factored <- df_ord %>% mutate(across(everything(), ordered)))
(df_bin_factored <- df_bin %>% mutate(across(everything(), ordered)))

# 5. Rebuild mixed-type data frame and clean
(df_mix2 <- bind_cols(df_cont, df_ord_factored, df_bin_factored))
(df_mix2_clean <- df_mix2[, colSums(is.na(df_mix2)) == 0])

# Debug: column classes and drop unsupported
cat(">>> DEBUG: column classes:\n")
print(sapply(df_mix2_clean, class))
allowed <- function(x) {
  inherits(x, c("numeric","integer","factor","ordered","logical","character"))
}
good_cols <- vapply(df_mix2_clean, allowed, logical(1))
bad_cols  <- names(good_cols)[!good_cols]
if (length(bad_cols) > 0) {
  cat(">>> WARNING: dropping unsupported columns:\n")
  print(bad_cols)
  df_mix2_clean <- df_mix2_clean[, good_cols]
}
cat(">>> POST‐CLEAN: column classes:\n")
print(sapply(df_mix2_clean, class))
(df_mix2_clean <- as.data.frame(df_mix2_clean))
cat("Post‐conversion class: ", class(df_mix2_clean), "\n")

# 6. Compute heterogeneous correlation matrix and compare eigenvalues
het_out   <- hetcor(df_mix2_clean, use = "pairwise.complete.obs")
R_mixed   <- het_out$correlations
stopifnot(!any(is.na(R_mixed)))

ev_raw <- eigen(hetcor(df_mix2_clean)$correlations)$values
ev_adj <- eigen(R_mixed)$values
plot(ev_raw, ev_adj, main = "Eigenvalue comparison")

# 7. Parallel analysis, MAP, and choose k
pa_out  <- fa.parallel(R_mixed, n.obs = nrow(df_mix2_clean),
                       fm = "minres", fa = "fa",
                       n.iter = 500, quant = .95,
                       cor = "cor", use = "pairwise",
                       plot = FALSE)
k_PA    <- pa_out$nfact
vss_out <- VSS(R_mixed, n = ncol(R_mixed),
               fm = "minres", n.obs = nrow(df_mix2_clean), plot = FALSE)
k_MAP   <- which.min(vss_out$map)
k       <- k_MAP  # choose as desired

# 8. Initial robust MINRES + oblimin
# number of observed variables and total loading‐vector length
p <- ncol(df_mix2_clean)
L <- p * k       # each replicate returns p×k loadings
B <- 1000

# 0. how many cores?
n_cores <- parallel::detectCores() - 1

# 1. spin up a SNOW cluster and register it
cl <- makeCluster(n_cores)
registerDoSNOW(cl)

# 2. make a text progress bar
pb       <- txtProgressBar(max = B, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)

# 3. pass that callback into foreach via .options.snow
opts <- list(progress = progress)

boot_loadings <- foreach(b = 1:B,
                         .combine      = rbind,
                         .packages     = c("psych", "polycor"),
                         .options.snow = opts) %dopar% {
  repeat {
    # a) draw a bootstrap sample
    idx <- sample(nrow(df_mix2_clean), replace = TRUE)
    d   <- df_mix2_clean[idx, , drop = FALSE]

    # b) mixed corr
    Rb <- tryCatch(
      hetcor(d, use = "pairwise.complete.obs")$correlations,
      error = function(e) NULL
    )
    if (is.null(Rb) || any(is.na(Rb))) next

    # c) EFA
    fa_rb <- tryCatch(
      fa(Rb,
         nfactors = k,
         fm       = "minres",
         rotate   = "oblimin",
         n.obs    = nrow(d)
      ),
      error = function(e) NULL
    )
    if (is.null(fa_rb)) next

    # d) success!
    Lvec  <- as.vector(fa_rb$loadings[])
    psi_rb <- fa_rb$uniquenesses
    return(c(Lvec, psi_rb))
  }
}

# 4. clean up
close(pb)
stopCluster(cl)
# separate out the psi‐columns (last p columns) and the lambda‐columns (first p*k)
lambda_boot <- boot_loadings[, 1:(p*k),           drop = FALSE]
psi_boot    <- boot_loadings[, (p*k + 1):(p*k + p), drop = FALSE]

# 6. compute medians and CIs
# — Λ
L_median <- matrix(apply(lambda_boot, 2, median), nrow = p, ncol = k)
L_ci     <- apply(lambda_boot, 2, quantile, c(.025, .975))
# — Ψ
psi_median <- apply(psi_boot, 2, median)
psi_ci     <- apply(psi_boot, 2, quantile, c(.025, .975))

# 7. name rows/cols
vars <- colnames(df_mix2_clean)
stopifnot(length(vars) == p)

rownames(L_median) <- vars
colnames(L_median) <- paste0("F", seq_len(k))
names(psi_median)  <- vars

# 8. reshape CI’s to long data.frames
ci_array   <- array(L_ci, dim = c(2, p, k))
ci_lower_L <- ci_array[1,,]
ci_upper_L <- ci_array[2,,]
df_L_ci <- data.frame(
  variable = rep(vars, times = k),
  factor   = rep(colnames(L_median), each = p),
  lower    = as.vector(ci_lower_L),
  upper    = as.vector(ci_upper_L),
  stringsAsFactors = FALSE
)
df_psi_ci <- data.frame(
  variable = vars,
  lower    = psi_ci[1, ],
  upper    = psi_ci[2, ],
  stringsAsFactors = FALSE
)

# 9. write to disk
write.csv(L_median,        "L_median.csv",   row.names = TRUE)
write.csv(df_L_ci,         "L_ci_long.csv",  row.names = FALSE)
write.csv(data.frame(variable = vars,
                     psi_median = psi_median),
          "psi_median.csv", row.names = FALSE)
write.csv(df_psi_ci,       "psi_ci.csv",     row.names = FALSE)

cat(sprintf("Wrote L_median, L_CI, psi_median and psi_CI to CSV.\n"))

# 1. Reload L_median (as a matrix, preserving row names)
L_median <- as.matrix(
  read.csv("L_median.csv", 
           row.names = 1,       # first column contains the row names
           check.names = FALSE  # so your original colnames aren’t munged
  )
)

# 2. Reload psi_median (as a named vector)
psi_df <- read.csv("psi_median.csv", 
                   stringsAsFactors = FALSE)

# extract the numeric vector
psi_median <- psi_df$psi_median

# re-assign the names
names(psi_median) <- psi_df$variable


# 4. Prune on the *refitted* communalities if you like
# comm <- rowSums(L_median^2)
# keep       <- names(comm)[comm >= 0.25]
# Lambda0    <- L_median[keep, , drop = FALSE]    # p_keep × k
# Psi0       <- psi_median[keep]                  # p_keep × 1
# R_prune    <- R_mixed[keep, keep]

# 4. Prune via decision‐tree -------------------------

# a) identify for each variable its primary factor and its 95% CI
vars        <- rownames(L_median)
# get median loadings matrix
Lmat        <- L_median
# long CI table already in df_L_ci
# compute per-variable primary factor, loading, and its CI
prim <- lapply(vars, function(v) {
  # subset CIs for this variable
  tmp <- df_L_ci[df_L_ci$variable == v, ]
  # pick the factor with max |median loading|
  loads <- Lmat[v, ]
  fidx  <- which.max(abs(loads))
  fi    <- colnames(Lmat)[fidx]
  med   <- loads[fidx]
  ci    <- tmp[tmp$factor == fi, c("lower","upper")]
  data.frame(
    variable    = v,
    primary_fac = fi,
    median_load = med,
    lower       = ci$lower,
    upper       = ci$upper,
    stringsAsFactors = FALSE
  )
})
prim_df <- do.call(rbind, prim)

# b) apply the decision‐rules
# 1) CI crosses zero?
prim_df$cross_zero <- with(prim_df, lower <= 0 & upper >= 0)
# 2) drop if crosses AND |median| < .30
drop1 <- prim_df$variable[prim_df$cross_zero & abs(prim_df$median_load) <  .30]
# 3) keep (but mark tentative) if crosses AND |median| ≥ .30
tent  <- prim_df$variable[prim_df$cross_zero & abs(prim_df$median_load) >= .30]
if(length(tent)) {
  cat("Mark tentative (CI crosses but |loading|≥.30):\n", paste(tent, collapse=", "), "\n\n")
}
# 4) all others survive
keep  <- setdiff(vars, drop1)
cat("Dropped for weak primary (CI crosses & |loading|<.30):\n", paste(drop1, collapse=", "), "\n\n")

# c) build pruned loading matrix and uniqueness vector
Lambda0 <- Lmat[keep, , drop = FALSE]
Psi0    <- psi_median[keep]

# d) now “constrain” trivial secondaries to zero:
#    for each row, find second‐largest loading and zero it if |ld|<.15
for(i in seq_len(nrow(Lambda0))) {
  row <- Lambda0[i, ]
  ord <- order(abs(row), decreasing = TRUE)
  # ord[1] is primary; ord[2] is secondary
  sec <- ord[2]
  if(abs(sec) < 0.15) {
    Lambda0[i, sec] <- 0
  }
}

# e) finalize pruned correlation sub‐matrix
R_prune <- R_mixed[keep, keep]


# ---------------------------------------------------
# 5. THEN prune any survivors with low communality
# ---------------------------------------------------
# compute communalities from the *constrained* loadings
h2_all <- rowSums(Lambda0^2)

# identify low‐communality items
drop_comm <- names(h2_all)[h2_all < 0.25]
if(length(drop_comm)) {
  cat("Dropping for low communality (h2<.25):", paste(drop_comm, collapse=", "), "\n")
}

# update keep‐list
keep_final <- setdiff(keep, drop_comm)

# rebuild pruned objects
Lambda0 <- Lambda0[keep_final, , drop = FALSE]
Psi0    <- Psi0[keep_final]
R_prune <- R_mixed[keep_final, keep_final]



# 2. Prepare storage for φ/H bootstrap...
# (rest of your workflow continues unchanged)


# 2. Prepare storage
B         <- 1000
k         <- ncol(Lambda0)
phis_rob  <- matrix(NA_real_, B, k)
Hs_rob    <- matrix(NA_real_, B, k)
completed <- 0
attempts  <- 0

# 2. set up cluster & progress bar
n_cores <- parallel::detectCores() - 1
cl      <- makeCluster(n_cores)
registerDoSNOW(cl)

pb       <- txtProgressBar(max = B, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts     <- list(progress = progress)

# 3. parallel bootstrap + compute φ and H
res <- foreach(b = 1:B,
               .combine    = rbind,
               .packages   = c("psych","polycor"),
               .options.snow = opts) %dopar% {
  repeat {
    # a) bootstrap sample of 'keep' cols
    samp_idx <- sample(nrow(df_mix2_clean), replace = TRUE)
    samp     <- df_mix2_clean[samp_idx, keep_final, drop = FALSE]

    # b) mixed correlation
    Rb <- tryCatch(
      hetcor(samp, use = "pairwise.complete.obs")$correlations,
      error = function(e) NULL
    )
    if (is.null(Rb) || any(is.na(Rb))) next

    # c) EFA
    fa_b <- tryCatch(
      fa(Rb,
         nfactors = k,
         fm       = "minres",
         rotate   = "oblimin",
         n.obs    = nrow(samp)
      ),
      error = function(e) NULL
    )
    if (is.null(fa_b)) next

    # d) compute φ and H
    Lb    <- fa_b$loadings
    phi_b <- diag(factor.congruence(Lambda0, Lb))
    uniqs <- 1 - rowSums(Lb[]^2)
    H_b   <- vapply(seq_len(k), function(j) {
      num <- sum(Lb[, j])^2
      num / (num + sum(uniqs))
    }, numeric(1))

    # success — return a single row of length 2*k
    return(c(phi_b, H_b))
  }
}

# 4. tear down
close(pb)
stopCluster(cl)

# 5. unpack results
phis_rob <- res[,        1:k,    drop = FALSE]
Hs_rob   <- res[, (k+1):(2*k),    drop = FALSE]

# 6. summarize
phi_mean <- colMeans(phis_rob)
H_mean   <- colMeans(Hs_rob)

cat(sprintf("Finished %d valid bootstraps\n", nrow(phis_rob)))
cat("Robust mean Tucker's φ: ", phi_mean, "\n")
cat("Robust mean Hancock's H:",  H_mean,   "\n")

# 15. Communalities & Residual Diagnostics ---

# 15.1 Communalities (h²) from Λ
h2 <- rowSums(Lambda0^2)
cat("Mean communality (h²):", mean(h2), "\n")
print(head(data.frame(variable = names(h2), communality = h2), 10))

# 15.2 Build uniqueness (Ψ) matrix correctly

# 1) Create the diagonal matrix from Psi0
Psi_mat <- diag(Psi0)

# 2) Assign row- and column names
rownames(Psi_mat) <- names(Psi0)
colnames(Psi_mat) <- names(Psi0)

# 15.3 Compute residual matrix: R_resid = R_prune − ΛΛᵀ − Ψ
resid_mat <- R_prune - (Lambda0 %*% t(Lambda0)) - Psi_mat

# 15.4 Overall fit: RMSR on the off-diagonals
off_diag_vals <- resid_mat[lower.tri(resid_mat)]
RMSR <- sqrt(mean(off_diag_vals^2))
cat("RMSR =", round(RMSR, 4), "\n")

# 15.5 Identify any |residual| > .10
off_idx <- which(abs(resid_mat) > 0.10, arr.ind = TRUE)
if (nrow(off_idx) > 0) {
  offenders <- data.frame(
    var1     = rownames(resid_mat)[off_idx[,1]],
    var2     = colnames(resid_mat)[off_idx[,2]],
    residual = resid_mat[off_idx]
  )
  cat("Residuals exceeding |.10|:\n")
  print(offenders)
} else {
  cat("No residuals exceed |0.10|.\n")
}

library(ggplot2)
library(reshape2)

# melt into long format
df_long <- melt(resid_mat, varnames = c("Row","Col"), value.name = "Residual")

ggplot(df_long, aes(x = Col, y = Row, fill = Residual)) +
  geom_tile() +
  scale_fill_gradient2(
    low     = "blue",
    mid     = "white",
    high    = "red",
    midpoint= 0
  ) +
  geom_text(aes(label = round(Residual, 2)), size = 2.5) +
  theme_minimal() +
  theme(
    axis.text.x  = element_text(angle = 45, hjust = 1, size = 8),
    axis.text.y  = element_text(size = 8),
    panel.grid   = element_blank()
  ) +
  labs(fill = "Residual")


# ---------------------------------------------------
# 16. Restrict all summaries & plots to the pruned items
# ---------------------------------------------------
library(dplyr)
library(tidyr)
library(ggplot2)

# 16.1: subset CI table to pruned variables
df_L_ci_pruned <- df_L_ci %>%
  filter(variable %in% keep_final)

# 16.2: subset median loadings matrix
L_median_pruned <- Lambda0
colnames(L_median_pruned) <- colnames(Lambda0)
rownames(L_median_pruned) <- keep_final

# ---------------------------------------------------
# 17. Summaries on pruned loadings
# ---------------------------------------------------
# unstable: CI includes zero
unstable_pruned <- df_L_ci_pruned %>%
  filter(lower <= 0 & upper >= 0)
cat("=== Pruned unstable loadings (CI spans 0) ===\n")
print(unstable_pruned)

# cross‐loadings: more than one factor with nonzero CI
cross_pruned <- df_L_ci_pruned %>%
  mutate(nonzero = (lower > 0 | upper < 0)) %>%
  filter(nonzero) %>%
  group_by(variable) %>%
  summarise(
    n_factors = n(),
    intervals = paste0(factor, "[", round(lower,2), ",", round(upper,2), "]", collapse = "; ")
  ) %>%
  filter(n_factors > 1)
cat("\n=== Pruned cross‐loading candidates ===\n")
print(cross_pruned)

# label candidates: CI entirely beyond ±0.30
label_pruned <- df_L_ci_pruned %>%
  filter(lower >= 0.30 | upper <= -0.30) %>%
  arrange(factor, desc(abs((lower+upper)/2)))
cat("\n=== Pruned labeling candidates (CI beyond ±0.30) ===\n")
print(label_pruned)

# ---------------------------------------------------
# 18. CI‐errorbar plot for pruned loadings
# ---------------------------------------------------
ggplot(df_L_ci_pruned,
       aes(x = reorder(variable, (lower+upper)/2),
           y = (lower+upper)/2, ymin = lower, ymax = upper,
           colour = factor)) +
  geom_errorbar(position = position_dodge(width = 0.6), width = 0.2) +
  geom_point(position = position_dodge(width = 0.6), size = 2) +
  coord_flip() +
  labs(
    x     = NULL,
    y     = "Median loading ±95% CI",
    title = "Pruned: Bootstrapped 95% CIs for Factor Loadings"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

# ---------------------------------------------------
# 19. Heatmap of pruned median loadings
# ---------------------------------------------------
L_df_pruned <- as.data.frame(L_median_pruned)
L_df_pruned$variable <- rownames(L_df_pruned)

L_long_pruned <- L_df_pruned %>%
  pivot_longer(-variable, names_to = "factor", values_to = "loading")

ggplot(L_long_pruned, aes(x = factor, y = variable, fill = loading)) +
  geom_tile() +
  scale_fill_gradient2(
    low      = "blue",
    mid      = "white",
    high     = "red",
    midpoint = 0,
    name     = "Loading"
  ) +
  labs(
    title = "Pruned: Median Factor Loadings Heatmap",
    x     = NULL,
    y     = NULL
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.text.y = element_text(size = 8),
    legend.position = "right"
  )

