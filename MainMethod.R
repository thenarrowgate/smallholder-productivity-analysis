# ---------------------------------------------------------------------------
# MainMethod.R
# ---------------------------------------------------------------------------
# This script performs the factor analytic pipeline described in the
# repository README.  After loading the cleaned Nepal or Senegal dataset it
# builds an appropriate correlation matrix, runs a large bootstrap EFA to
# obtain stable loadings, prunes unstable items and finally assesses the
# solution's robustness via Tucker's phi and Hancock's H.
# ---------------------------------------------------------------------------

# Step 1 ─ Load required packages
library(dplyr)       # data wrangling
library(EFAtools)    # VSS(), tenBerge scores
library(boot)        # bootstrap()
library(Gifi)        # princals()
library(lavaan)      # sem()
library(semTools)    # indProd() for latent interactions
library(mgcv)        # gam()
library(polycor)     # hetcor()
library(psych)       # mixedCor(), fa.*, factor.congruence(), factor.scores
library(readxl)      # read_excel()
library(stringr)     # str_split()
library(doSNOW)      # parallel backend
library(foreach)     # foreach looping
library(ggplot2)     # plotting
library(reshape2)    # melt()
library(tidyr)       # pivot_longer()
library(WGCNA)   # provides bicor()
library(Matrix)   # nearPD for KMO/Bartlett

# Step 2 ─ Set seed and working directory
# Resolve where the script is running from so relative paths work both when
# executed interactively and via Rscript.  A seed is set for reproducible
# bootstrap samples.
set.seed(2025)
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) {
  LOCAL_DIR <- args[1]
} else {
  full_args <- commandArgs(trailingOnly = FALSE)
  file_idx <- grep("^--file=", full_args)
  if (length(file_idx) > 0) {
    script_path <- sub("^--file=", "", full_args[file_idx[1]])
    LOCAL_DIR <- dirname(normalizePath(script_path))
  } else if (requireNamespace("here", quietly = TRUE)) {
    LOCAL_DIR <- here::here()
  } else {
    LOCAL_DIR <- "."
  }
}
setwd(LOCAL_DIR)

# Step 3 ─ Load the prepared dataset
# The Python preprocessing notebook saves a clean Excel file. The outcome
# columns used later in SEM are removed here so that EFA only sees the
# predictor variables.
df <- read_excel("nepal_dataframe_FA.xlsx")
y_prod <- df$Q0__AGR_PROD__continuous
df     <- df %>% select(-Q0__AGR_PROD__continuous,
                        -Q0__sustainable_livelihood_score__continuous)

# Step 4 ─ Split variables by declared type
# Variable names encode their measurement level after the final "__".
# This split lets us treat each class appropriately when forming the
# correlation matrix.
types <- str_split(names(df), "__", simplify = TRUE)[,3]
types[types == "binary_nominal"] <- "nominal"
df_cont <- df[, types == "continuous", drop = FALSE]
df_ord  <- df[, types == "ordinal",    drop = FALSE]
df_bin  <- df[, types == "binary",     drop = FALSE]
df_nom  <- df[, types == "nominal",    drop = FALSE]

# Step 5 ─ Convert ordinal/binary variables to ordered factors
# Treating these as ordered ensures that polychoric/polyserial correlations
# are used later when ``COR_METHOD == "mixed"``.
df_ord_factored <- df_ord %>% mutate(across(everything(), ordered))
df_bin_factored <- df_bin %>% mutate(across(everything(), ordered))

# Step 6 ─ Reassemble a single data frame and remove incomplete cases
# Only columns with no missing values are kept for EFA.
df_mix2       <- bind_cols(df_cont, df_ord_factored, df_bin_factored)
df_mix2_clean <- df_mix2[, colSums(is.na(df_mix2)) == 0]

# Step 7 ─ Sanity check the columns
# Print each column's class and drop anything unexpected.  In practice all
# variables should be numeric, integer, factor or ordered.  This guard helps
# catch stray list-columns from earlier processing.
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


# ── Step 8 ─ Compute correlation matrix ---------------------------------------
# ``COR_METHOD`` toggles between a heterogeneous polychoric/polyserial
# correlation matrix ("mixed") and a simpler Spearman matrix treating all
# variables as numeric ranks.  The mixed option is slower but more faithful
# to the measurement levels.
COR_METHOD <- "spearman"

if (COR_METHOD == "mixed") {
  het_out <- hetcor(df_mix2_clean, use = "pairwise.complete.obs")
  R_mixed <- het_out$correlations   # base: polyserial/polychoric + Pearson from hetcor
} else if (COR_METHOD == "spearman") {
  df_numeric <- df_mix2_clean %>% mutate(across(everything(), as.numeric))
  R_mixed <- cor(df_numeric, method = "spearman", use = "pairwise.complete.obs")
} else {
  stop("Unknown COR_METHOD")
}

# --- REBUILD type vector to align with df_mix2_clean ---------------------------
# (We can't reuse `types` from Step 4 because columns were dropped.)
types_clean <- str_split(names(df_mix2_clean), "__", simplify = TRUE)[, 3]
types_clean[types_clean == "binary_nominal"] <- "nominal"  # just in case

# --- Identify continuous cols in the CLEANED data ------------------------------
cont_idx <- which(types_clean == "continuous")

# (Optional) quick report
cat("Continuous vars in cleaned data:", length(cont_idx), "\n")

# --- Robust bicor for continuous block ----------------------------------------
if (COR_METHOD == "mixed" && length(cont_idx) > 1) {
  
  # Extract continuous subset from the *cleaned* data
  df_cont_clean <- df_mix2_clean[, cont_idx, drop = FALSE]
  
  # Optional: flag zero-MAD columns (can distort bicor)
  mad_vals <- apply(df_cont_clean, 2, mad, na.rm = TRUE)
  if (any(mad_vals == 0)) {
    cat("WARNING: zero MAD in continuous vars:\n")
    print(names(mad_vals)[mad_vals == 0])
    # bicor() will fallback to Pearson for these vars; acceptable unless many.
  }
  
  # Compute robust correlation
  R_cont <- WGCNA::bicor(
    as.matrix(df_cont_clean),
    use = "pairwise.complete.obs"
  )
  
  # Overwrite the continuous-continuous block in the mixed matrix
  R_mixed[cont_idx, cont_idx] <- R_cont
  
  # Enforce symmetry numerically (defensive; should already be symmetric)
  R_mixed[cont_idx, cont_idx] <-
    (R_mixed[cont_idx, cont_idx] + t(R_mixed[cont_idx, cont_idx])) / 2
  
  # Force exact 1s on the diagonal (good hygiene before EFA)
  diag(R_mixed) <- 1
}

# Regardless of method we ensure the matrix is positive definite before
# KMO/Bartlett tests or factor extraction. nearPD performs a minimal
# adjustment when needed.
R_mixed <- as.matrix(nearPD(R_mixed, corr = TRUE)$mat)

stopifnot(!any(is.na(R_mixed)))
# Step 8b - Suitability checks: KMO and Bartlett tests
# These diagnostics help verify whether factor analysis is appropriate for
# the correlation matrix.
kmo_res <- psych::KMO(R_mixed)
bart_res <- psych::cortest.bartlett(R_mixed, n = nrow(df_mix2_clean))
cat("KMO overall MSA:", round(kmo_res$MSA, 3), "\n")
cat("Bartlett's test p-value:", signif(bart_res$p.value, 3), "\n")

# -- Per-variable MSA assessment ----------------------------------------------
# Drop variables that fall below the commonly-used MSA threshold of 0.5.
msa_vec <- kmo_res$MSAi
names(msa_vec) <- colnames(R_mixed)
low_msa <- msa_vec[msa_vec < 0.5]
if (length(low_msa) > 0) {
  cat("Variables dropped for low MSA (<0.5):\n")
  print(round(low_msa, 3))
  keep_vars <- setdiff(colnames(R_mixed), names(low_msa))
  R_mixed <- R_mixed[keep_vars, keep_vars]
  df_mix2_clean <- df_mix2_clean[, keep_vars, drop = FALSE]
} else {
  cat("No variables dropped for low MSA\n")
}


if (COR_METHOD == "mixed") {
  ev_raw <- eigen(hetcor(df_mix2_clean, use = "pairwise.complete.obs")$correlations)$values
} else {
  df_num_ev <- as.data.frame(lapply(df_mix2_clean, as.numeric))
  ev_raw <- eigen(cor(df_num_ev, method = "spearman", use = "pairwise.complete.obs"))$values
}
ev_adj <- eigen(R_mixed)$values
# Quick diagnostic plot to ensure the adjusted correlation matrix retains a
# similar eigen spectrum to the raw correlations.
plot(ev_raw, ev_adj, main="Eigenvalue comparison")



# Step 9 ─ Determine number of factors (parallel analysis & MAP)
# Two common heuristics are used: Horn's parallel analysis (upper bound)
# and Velicer's MAP (lower bound).  ``k`` can be set manually or using one
# of these estimates.
pa_out <- fa.parallel(R_mixed, n.obs = nrow(df_mix2_clean),
                      fm = "minres", fa = "fa",
                      n.iter = 500, quant = .95,
                      cor = "cor", use = "pairwise", plot = FALSE)
k_PA  <- pa_out$nfact
vss_out <- VSS(R_mixed, n = ncol(R_mixed),
               fm = "minres", n.obs = nrow(df_mix2_clean), plot = FALSE)
k_MAP <- which.min(vss_out$map)
k     <- 2 # manual override; could also use k_MAP or k_PA

# Step 10 ─ Bootstrap robust MINRES+geomin to get loadings & uniquenesses
# Each iteration draws a bootstrap sample, computes a correlation matrix and
# extracts a factor solution.  The median across iterations serves as a robust
# estimate of the loadings.
p <- ncol(df_mix2_clean)
B <- 1000
n_cores <- max(1, parallel::detectCores() - 1)
cl <- makeCluster(n_cores); registerDoSNOW(cl)
pb <- txtProgressBar(max=B, style=3)
opts <- list(progress = function(n) setTxtProgressBar(pb, n))

boot_load <- foreach(b=1:B, .combine=rbind,
                     .packages=c("psych","polycor","Matrix"),
                     .options.snow=opts) %dopar% {
                       repeat {
                         samp <- df_mix2_clean[sample(nrow(df_mix2_clean), replace=TRUE), ]
                         Rb   <- tryCatch({
                           if (COR_METHOD == "mixed") {
                             hetcor(samp, use = "pairwise.complete.obs")$correlations
                           } else {
                             samp_num <- as.data.frame(lapply(samp, as.numeric))
                             cor(samp_num, method = "spearman", use = "pairwise.complete.obs")
                           }
                         }, error=function(e) NULL)
                         if(is.null(Rb) || any(is.na(Rb))) next
                         # Stabilise the correlation matrix for "fa" in case
                         # bootstrapping produced a non positive-definite Rb
                         Rb   <- as.matrix(nearPD(Rb, corr = TRUE)$mat)
                         fa_b <- tryCatch(fa(Rb, nfactors=k, fm="minres", rotate="geominQ", n.obs=nrow(samp)),
                                          error=function(e) NULL)
                         if(is.null(fa_b)) next
                         return(c(as.vector(fa_b$loadings[]), fa_b$uniquenesses))
                       }
                     }
close(pb); stopCluster(cl)

# Step 11 ─ Summarize bootstrap results
# The distributions of loadings and uniquenesses are summarised by their
# medians and 95% confidence intervals.  These will later be used for
# pruning unstable items.
lambda_boot <- boot_load[, 1:(p*k)]
psi_boot    <- boot_load[, (p*k+1):(p*k+p)]
L_median    <- matrix(apply(lambda_boot, 2, median), nrow=p, ncol=k)
L_ci        <- apply(lambda_boot, 2, quantile, c(.025,.975))
psi_median  <- apply(psi_boot, 2, median)
# name dimensions
vars <- colnames(df_mix2_clean)
rownames(L_median) <- vars
colnames(L_median) <- paste0("F", 1:k)
names(psi_median)  <- vars
# reshape CIs
ci_arr     <- array(L_ci, dim=c(2,p,k))
df_L_ci    <- data.frame(
  variable = rep(vars, each=k),
  factor   = rep(colnames(L_median), times=p),
  lower    = as.vector(ci_arr[1,,]),
  upper    = as.vector(ci_arr[2,,]),
  stringsAsFactors=FALSE
)

# build df_psi_ci for the uniqueness‐CI
psi_ci   <- apply(psi_boot, 2, quantile, c(.025, .975))
df_psi_ci <- data.frame(
  variable = vars,
  lower    = psi_ci[1, ],
  upper    = psi_ci[2, ],
  stringsAsFactors = FALSE
)


# The bootstrapped summaries remain in memory so they can be used directly
# without the intermediate CSV round trip.  ``L_median`` and ``df_L_ci`` hold
# the factor loading medians and confidence intervals, while ``psi_median`` and
# ``df_psi_ci`` contain the uniqueness summaries.  These objects will feed into
# the pruning step below.

# Step 12 ─ Prune items via decision-tree rules
# Variables with unstable or weak loadings are removed to improve factor
# interpretability.  Primary loadings are defined along with their
# bootstrapped confidence intervals.
#   12.1 Identify each variable’s primary loading & its 95% CI
prim_list <- lapply(vars, function(v) {
  tmp <- df_L_ci[df_L_ci$variable==v, ]
  loads <- L_median[v,]; fidx <- which.max(abs(loads))
  data.frame(
    variable    = v,
    primary_fac = names(loads)[fidx],
    median_load = loads[fidx],
    lower       = tmp$lower[tmp$factor==names(loads)[fidx]],
    upper       = tmp$upper[tmp$factor==names(loads)[fidx]],
    stringsAsFactors=FALSE
  )
})
prim_df <- do.call(rbind, prim_list)
#   12.2 Apply rules: drop if CI crosses 0 AND |median|<.30; mark tentative if ≥.30
prim_df$cross_zero <- with(prim_df, lower<=0 & upper>=0)
drop1 <- prim_df$variable[prim_df$cross_zero & abs(prim_df$median_load)<.30]
tent  <- prim_df$variable[prim_df$cross_zero & abs(prim_df$median_load)>=.30]
if(length(tent)) message("Tentative (cross-zero but |load|≥.30): ", paste(tent, collapse=", "))
keep  <- setdiff(vars, drop1)
message("Dropped (cross-zero & |load|<.30): ", paste(drop1, collapse=", "))

#   12.3 Build pruned Λ and Ψ
Lambda0 <- L_median[keep, , drop=FALSE]
Psi0    <- psi_median[keep]

#   12.4 Zero‐out trivial secondaries (<.15)
if(ncol(Lambda0) > 1) {
  for(i in seq_len(nrow(Lambda0))) {
    row <- Lambda0[i,]; idx <- order(abs(row), decreasing=TRUE)
    sec <- idx[2]
    if(abs(row[sec]) < .15) Lambda0[i, sec] <- 0
  }
}

R_prune <- R_mixed[keep, keep]

# Step 13 ─ Final pruning based on communality
# Items whose communality falls below 0.30 after the previous steps are
# removed.  The resulting correlation matrix feeds into a final bootstrap
# to compute Tucker's phi and Hancock's H.
h2   <- rowSums(Lambda0^2)
drop_comm <- names(h2)[h2<0.3]
if(length(drop_comm)) message("Dropping low-h² (<.30): ", paste(drop_comm, collapse=", "))
keep_final <- setdiff(keep, drop_comm)
Lambda0    <- Lambda0[keep_final, , drop=FALSE]
Psi0       <- Psi0[keep_final]
R_prune    <- R_mixed[keep_final, keep_final]

# (… continue φ/H bootstrap, residual diagnostics, plotting …)
B         <- 1000
k         <- ncol(Lambda0)
phis_rob  <- matrix(NA_real_, B, k)
Hs_rob    <- matrix(NA_real_, B, k)
completed <- 0
attempts  <- 0

# set up cluster & progress bar
n_cores <- max(1, parallel::detectCores() - 1)
cl      <- makeCluster(n_cores)
registerDoSNOW(cl)

pb       <- txtProgressBar(max = B, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts     <- list(progress = progress)

# parallel bootstrap + compute φ and H
# Each iteration refits the factor model on a bootstrap sample and computes
# Tucker's congruence with the target solution as well as Hancock's H.
res <- foreach(b = 1:B,
               .combine    = rbind,
               .packages   = c("psych","polycor","Matrix"),
               .options.snow = opts) %dopar% {
                 repeat {
                   # a) bootstrap sample of 'keep' cols
                   samp_idx <- sample(nrow(df_mix2_clean), replace = TRUE)
                   samp     <- df_mix2_clean[samp_idx, keep_final, drop = FALSE]
                   
                   # b) correlation
                   Rb <- tryCatch(
                     {
                       if (COR_METHOD == "mixed") {
                         hetcor(samp, use = "pairwise.complete.obs")$correlations
                       } else {
                         samp_num <- as.data.frame(lapply(samp, as.numeric))
                         cor(samp_num, method = "spearman", use = "pairwise.complete.obs")
                       }
                     },
                     error = function(e) NULL
                   )
                   if (is.null(Rb) || any(is.na(Rb))) next
                   # Stabilise Rb so ``fa`` does not fail due to non positive-definiteness
                   Rb <- as.matrix(nearPD(Rb, corr = TRUE)$mat)
                   
                   # c) EFA
                   fa_b <- tryCatch(
                     fa(Rb,
                        nfactors = k,
                        fm       = "minres",
                        rotate   = "geominQ",
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

# tear down
close(pb)
stopCluster(cl)

# 5. unpack results
phis_rob <- res[,        1:k,    drop = FALSE]
Hs_rob   <- res[, (k+1):(2*k),    drop = FALSE]

# 6. summarize
# Average across successful bootstraps
phi_mean <- colMeans(phis_rob)
H_mean   <- colMeans(Hs_rob)

cat(sprintf("Finished %d valid bootstraps\n", nrow(phis_rob)))
cat("Robust mean Tucker's φ: ", phi_mean, "\n")
cat("Robust mean Hancock's H:",  H_mean,   "\n")

# ---------------------------------------------------------------------------
# 15. Communalities & Residual Diagnostics
# ---------------------------------------------------------------------------
# Compute item communalities from the pruned loading matrix.  These give a
# quick check of how well each retained variable is represented by the
# extracted factors.
h2 <- rowSums(Lambda0^2)
cat("Mean communality (h²):", mean(h2), "\n")
print(head(data.frame(variable = names(h2), communality = h2), 10))

# Build the uniqueness matrix Ψ from the vector of uniquenesses.  Having a
# proper diagonal matrix makes later residual calculations clearer.
Psi_mat <- diag(Psi0)
rownames(Psi_mat) <- names(Psi0)
colnames(Psi_mat) <- names(Psi0)

# Residual correlation matrix: R_resid = R_prune − ΛΛᵀ − Ψ
resid_mat <- R_prune - (Lambda0 %*% t(Lambda0)) - Psi_mat

# Overall misfit measured by the root-mean-square residual (RMSR) on the
# off-diagonal elements only.
off_diag_vals <- resid_mat[lower.tri(resid_mat)]
RMSR <- sqrt(mean(off_diag_vals^2))
cat("RMSR =", round(RMSR, 4), "\n")

# Identify any residuals with absolute value greater than 0.10, which can
# highlight local areas of poor fit.
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

# Visualise the residual matrix with a heatmap for an at-a-glance view of
# where correlations are over- or under-estimated by the factor model.
df_long <- reshape2::melt(resid_mat, varnames = c("Row", "Col"),
                          value.name = "Residual")
ggplot(df_long, aes(x = Col, y = Row, fill = Residual)) +
  geom_tile() +
  scale_fill_gradient2(
    low      = "blue",
    mid      = "white",
    high     = "red",
    midpoint = 0
  ) +
  geom_text(aes(label = round(Residual, 2)), size = 2.5) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    axis.text.y = element_text(size = 8),
    panel.grid  = element_blank()
  ) +
  labs(fill = "Residual")

# ---------------------------------------------------------------------------
# 16. Restrict all summaries & plots to the pruned items
# ---------------------------------------------------------------------------
df_L_ci_pruned <- df_L_ci %>%
  filter(variable %in% keep_final)

L_median_pruned <- Lambda0
colnames(L_median_pruned) <- colnames(Lambda0)
rownames(L_median_pruned) <- keep_final

# ---------------------------------------------------------------------------
# 17. Summaries on pruned loadings
# ---------------------------------------------------------------------------
# Variables whose confidence interval spans zero are considered unstable.
unstable_pruned <- df_L_ci_pruned %>%
  filter(lower <= 0 & upper >= 0)
cat("=== Pruned unstable loadings (CI spans 0) ===\n")
print(unstable_pruned)

# Identify cross-loading candidates: variables with non-zero CIs on more than
# one factor.
cross_pruned <- df_L_ci_pruned %>%
  mutate(nonzero = (lower > 0 | upper < 0)) %>%
  filter(nonzero) %>%
  group_by(variable) %>%
  summarise(
    n_factors = n(),
    intervals = paste0(factor, "[", round(lower, 2), ",", round(upper, 2), "]",
                       collapse = "; ")
  ) %>%
  filter(n_factors > 1)
cat("\n=== Pruned cross-loading candidates ===\n")
print(cross_pruned)

# Candidate labels: loadings whose entire CI lies beyond ±0.30.
label_pruned <- df_L_ci_pruned %>%
  filter(lower >= 0.30 | upper <= -0.30) %>%
  arrange(factor, desc(abs((lower + upper) / 2)))
cat("\n=== Pruned labeling candidates (CI beyond ±0.30) ===\n")
print(label_pruned)

# ---------------------------------------------------------------------------
# 18. CI-errorbar plot for pruned loadings
# ---------------------------------------------------------------------------
ggplot(df_L_ci_pruned,
       aes(x = reorder(variable, (lower + upper) / 2),
           y = (lower + upper) / 2, ymin = lower, ymax = upper,
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

# ---------------------------------------------------------------------------
# 19. Heatmap of pruned median loadings
# ---------------------------------------------------------------------------
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
    axis.text.x    = element_text(angle = 45, hjust = 1),
    axis.text.y    = element_text(size = 8),
    legend.position = "right"
  )

# ---------------------------------------------------------------------------
# 20. Factor scores and GAM regression
# ---------------------------------------------------------------------------
# After pruning we compute Ten Berge factor scores for the retained variables
# and examine their relationship with the productivity index using a
# generalised additive model.  Nominal predictors are collapsed to an "Other"
# level if a category represents fewer than 5% of observations.

# --- 20.1  Compute Thurstone scores for the final items ---------------------raw_data <- df_mix2_clean[, keep_final, drop = FALSE]

# 1.  Prepare the data matrix exactly as you did before
raw_data <- df_mix2_clean[, keep_final, drop = FALSE]

raw_data_num <- data.frame(lapply(raw_data, function(col) {
  if (is.numeric(col)) {
    col
  } else if (is.factor(col) || is.ordered(col)) {
    as.numeric(col)
  } else {
    stop("Unsupported column class: ", class(col))
  }
}))

# 2.  Z-score the observed variables                           (n × p)
X_std <- scale(raw_data_num)

# 3.  Build Ψ⁻¹ once – Psi0 is your vector of uniquenesses     (p × p)
Psi_inv <- diag(1 / Psi0)

# 4.  Bartlett weight matrix:  W_B = Ψ⁻¹ Λ (Λᵀ Ψ⁻¹ Λ)⁻¹        (p × k)
middle  <- t(Lambda0) %*% Psi_inv %*% Lambda0      # (k × k)
W_B     <- Psi_inv %*% Lambda0 %*% solve(middle)   # (p × k)

# 5.  Factor scores (these *retain* the F-to-F correlations)   (n × k)
F_hat <- X_std %*% W_B

# 6.  Keep the same labels so downstream code is unchanged
colnames(F_hat) <- colnames(Lambda0)   # “F1”, “F2”, …
rownames(F_hat) <- rownames(df_mix2_clean)

# --- 20.2  Prepare nominal predictors --------------------------------------
collapse_rare <- function(f, threshold = 0.05) {
  tab <- prop.table(table(f))
  rare <- names(tab)[tab < threshold]
  f_new <- as.character(f)
  f_new[f_new %in% rare] <- "Other"
  factor(f_new)
}

nom_collapsed <- df_nom %>% mutate(across(everything(), collapse_rare))
is_large <- function(v) nlevels(v) > 10
large_noms <- names(Filter(is_large, nom_collapsed))
small_noms <- names(Filter(Negate(is_large), nom_collapsed))

# --- 20.3  Fit GAM of productivity on factor scores ------------------------
gam_df <- data.frame(prod_index = y_prod, F_hat, nom_collapsed)
gam_df <- na.omit(gam_df)

smooth_terms <- paste0("s(", colnames(F_hat), ")")
small_terms  <- if (length(small_noms)) paste(small_noms, collapse = " + ") else NULL
large_terms  <- if (length(large_noms)) paste0("s(", large_noms, ", bs='re')", collapse = " + ") else NULL
all_terms <- c(smooth_terms, small_terms, large_terms)
gam_form <- as.formula(paste("prod_index ~", paste(all_terms, collapse = " + ")))

gam_fit <- mgcv::gam(gam_form, data = gam_df, method = "REML", select = TRUE)
print(summary(gam_fit)$s.table)
print(summary(gam_fit)$p.table)

par(mfrow = c(1, ncol(F_hat)))
plot(gam_fit, pages = 1, all.terms = TRUE, shade = TRUE)

# ---- Additional diagnostics for the GAM fit ----
# 1. Overall deviance explained (R^2)
dev_expl <- summary(gam_fit)$dev.expl
cat("Deviance explained:", round(dev_expl, 3), "\n")

# 2. Concurvity check
concurv <- mgcv::concurvity(gam_fit)
print(concurv)

# 3. Residual plots
par(mfrow = c(1, 1))
plot(gam_fit, residuals = TRUE)

param_terms <- summary(gam_fit)$pTerms.table
if (!is.null(param_terms) && nrow(param_terms) > 0) {
  p_adj <- p.adjust(param_terms[, "p-value"], method = "fdr")
  print(p_adj)
  signif_terms <- rownames(param_terms)[p_adj < 0.05]
} else {
  p_adj <- numeric(0)
  signif_terms <- character(0)
}

# 5. Refit using only significant parametric terms and compare AIC
refit_terms <- c(smooth_terms, signif_terms, large_terms)
if (length(refit_terms) == 0) {
  refit_form <- as.formula("prod_index ~ 1")
} else {
  refit_form <- as.formula(paste("prod_index ~", paste(refit_terms, collapse = " + ")))
}
gam_refit <- mgcv::gam(refit_form, data = gam_df, method = "REML", select = TRUE)
cat("AIC(original)=", AIC(gam_fit), " AIC(refit)=", AIC(gam_refit), "\n")

print(summary(gam_refit)$s.table)
print(summary(gam_refit)$p.table)

par(mfrow = c(1, ncol(F_hat)))
plot(gam_refit, pages = 1, all.terms = TRUE, shade = TRUE)

# Check basis dimension adequacy
gam.check(gam_refit)          # k-index; want > ~0.9 and p>0.05

# Concurvity reduced?
mgcv::concurvity(gam_refit)

# Compare nested models formally
anova(gam_fit, gam_refit, test="Chisq")  # uses ML; may need method="ML" refits

# Compute the raw correlation between the two factor scores
cor(F_hat[, 1], F_hat[, 2])

# Test whether F₂ relates to productivity before controlling for F₁
cor(F_hat)
m_F2 <- mgcv::gam(y_prod ~ s(F_hat[, 2]), method = "REML")

summary(m_F2)      # gives the p-value and % deviance explained
anova(m_F2, gam_refit, test="Chisq")  # formally compare to a model that already has F1
plot(m_F2, shade=TRUE)             # visualise the smooth

# ------------------------------------------------------------------
# 1.  Make sure the factor scores sit in the data frame
# ------------------------------------------------------------------
gam_df$F1 <- F_hat[, 1]   # commercial / management intensity
gam_df$F2 <- F_hat[, 2]   # wealth & psycho-social resources

# ------------------------------------------------------------------
# 2A.  **Binary varying-coefficient** version
#      — lets the F1 smooth have a different shape for the top-F2
#        households ( >  +1 SD ) versus the rest.
# ------------------------------------------------------------------
zF2          <- scale(gam_df$F2)[, 1]             # z-score for convenience
gam_df$F2_hi <- factor(ifelse(zF2 > 1, "High", "Other"))

m_vc <- mgcv::gam(
  prod_index ~
    s(F1) +                         # baseline F1 smooth
    s(F1, by = F2_hi) +             # deviation for high-F2 farms
    s(F2)                           # main (residual) effect of F2
  , data   = gam_df
  , method = "REML"
)

cat("\n--- Varying-coefficient model (binary high-F2) ---\n")
print(summary(m_vc))

# Visualise: two over-laid F1 curves
plot(m_vc, select = 1, main = "F1 effect: baseline (Other F2)")      # s(F1)
plot(m_vc, select = 2, add = TRUE, col = 2)                          # s(F1)·HighF2
legend("topleft", legend = c("Other F2", "High F2"), lwd = 2, col = 1:2)

# Formal comparison with your previous refit that already contained F1
anova(gam_refit, m_vc, test = "Chisq")


# ------------------------------------------------------------------
# 2B.  **Continuous tensor-product surface** version
#      — estimates a smooth response surface f(F1, F2).
# ------------------------------------------------------------------
m_te <- mgcv::gam(
  prod_index ~ te(F1, F2)           # 2-D smooth
  , data   = gam_df
  , method = "REML"
)

cat("\n--- Tensor interaction model (continuous F1 × F2) ---\n")
print(summary(m_te))

# Perspective or contour plot
vis.gam(m_te, view = c("F1", "F2"), plot.type = "persp",
        color = "heat", main = "Productivity surface f(F1, F2)")

# Compare against an additive model with no interaction
m_add <- mgcv::gam(prod_index ~ s(F1) + s(F2), data = gam_df, method = "REML")
anova(m_add, m_te, test = "Chisq")

# ---------------------------------------------------------------------------
# 21. Confirmatory factor analysis and SEM
# ---------------------------------------------------------------------------
# Use the pruned loading matrix ``Lambda0`` to build a lavaan model
# where each variable loads only on its primary factor.  The structural
# part regresses the productivity index on the two latent factors and
# their interaction, reflecting the synergy observed in the GAM phase.

# --- 21.1  Assemble measurement model from primary loadings ---------------
prim_fac <- apply(abs(Lambda0), 1, function(row)
  colnames(Lambda0)[which.max(row)])
items_by_fac <- split(names(prim_fac), prim_fac)
meas_lines <- vapply(names(items_by_fac), function(f)
  paste0(f, " =~ ", paste(items_by_fac[[f]], collapse = " + ")),
  character(1))

# --- 21.2  Build full SEM specification ----------------------------------
# Create product indicators so the latent interaction F1 × F2 can be modelled
# (lavaan does not directly support the ":" operator for latent factors).
df_sem <- df_mix2_clean[, keep_final, drop = FALSE]
df_sem$prod_index <- y_prod

ordered_vars <- names(df_sem)[sapply(df_sem, is.ordered)]

# Build product indicators via semTools (needs numeric columns)
prod_base <- df_sem
prod_base[ordered_vars] <- lapply(prod_base[ordered_vars], as.numeric)
num_cols_all <- names(prod_base)[sapply(prod_base, is.numeric)]
prod_base[num_cols_all] <- scale(prod_base[num_cols_all])
nprod <- length(items_by_fac[["F1"]]) * length(items_by_fac[["F2"]])
prod_names <- paste0("F1F2_", seq_len(nprod))
df_sem <- semTools::indProd(
  data   = prod_base,
  var1   = items_by_fac[["F1"]],
  var2   = items_by_fac[["F2"]],
  match  = FALSE,
  meanC  = TRUE,
  namesProd = prod_names
)
int_indicators <- setdiff(colnames(df_sem), colnames(prod_base))
df_sem[ordered_vars] <- lapply(df_sem[ordered_vars], ordered)

# Measurement model including the interaction factor
meas_int_lines <- c(
  meas_lines,
  paste0("F1F2 =~ ", paste(int_indicators, collapse = " + "))
)

# Full SEM specification with the latent interaction factor regressing prod_index
sem_lines <- c(meas_int_lines,
               "prod_index ~ F1 + F2 + F1F2")
sem_model <- paste(sem_lines, collapse = "\n")
cat("\nSEM model specification:\n", sem_model, "\n")

# --- 21.3  Fit CFA model --------------------------------------------------
fit_cfa <- lavaan::cfa(paste(meas_int_lines, collapse = "\n"),
                       data    = df_sem,
                       ordered = ordered_vars,
                       std.lv  = TRUE)
cat("\n--- CFA summary ---\n")
print(summary(fit_cfa, fit.measures = TRUE, standardized = TRUE))

# --- 21.5  Fit SEM with latent interaction -------------------------------
fit_sem <- lavaan::sem(sem_model,
                       data    = df_sem,
                       ordered = ordered_vars,
                       std.lv  = TRUE)
cat("\n--- SEM summary ---\n")
print(summary(fit_sem, fit.measures = TRUE, standardized = TRUE))

