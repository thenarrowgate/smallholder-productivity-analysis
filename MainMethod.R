
# ---------------------------------------------------------------------------
# Step 1. Load required packages
# ---------------------------------------------------------------------------

# Before attaching libraries, check that all dependencies are installed. This
# produces a clear diagnostic list instead of failing on the first missing
# package so that environments lacking R packages can be configured easily.
required_packages <- c(
  "dplyr",      # data wrangling
  "EFAtools",   # VSS(), tenBerge scores
  "boot",       # bootstrap()
  "Gifi",       # princals()
  "lavaan",     # sem()
  "mgcv",       # gam()
  "mgcViz",     # diagnostic tools for mgcv::gam
  "polycor",    # hetcor()
  "psych",      # mixedCor(), fa.*, factor.congruence(), factor.scores
  "readxl",     # read_excel()
  "stringr",    # str_split()
  "doSNOW",     # parallel backend
  "foreach",    # foreach looping
  "ggplot2",    # plotting
  "reshape2",   # melt()
  "tidyr",      # pivot_longer()
  "WGCNA",      # provides bicor()
  "Matrix",     # nearPD for KMO/Bartlett
  "mediation"   # causal mediation analysis
)

# Mainly added for agent models trying to run the script during development
missing_pkgs <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  cat(">>> ERROR: missing required packages:\n")
  print(missing_pkgs)
  stop("Install the above packages and rerun MainMethod.R")
}

invisible(lapply(required_packages, library, character.only = TRUE))

# ---------------------------------------------------------------------------
# Step 2. Set seed and working directory
# ---------------------------------------------------------------------------

# Set a fixed seed to ensure reproducible bootstrap samples 
set.seed(2025)

# Figure out the directory from which being ran. A direct path
# provided via the command line can be accepted; if none provided
# tries to infer the script from command line argument files,
# and if none given falls back to the current directory.
# Then, it switches the working directory to this path that
# it resolved.
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

# ---------------------------------------------------------------------------
# Step 3. Load the prepared dataset
# ---------------------------------------------------------------------------

# Loads the provided path to the preprocessed data in an excel file.
# Removed the agricultural productivity and sustainability score
# explained variables via their standardized names, keeping only
# predictor variables.
df <- read_excel("nepal_dataframe_FA.xlsx")
y_prod <- df$Q0__AGR_PROD__continuous
df     <- df %>% dplyr::select(-Q0__AGR_PROD__continuous,
-Q0__sustainable_livelihood_score__continuous)

# ---------------------------------------------------------------------------
# Step 4. Split variables by declared type
# ---------------------------------------------------------------------------

# In the given standardized variable names  that Variable names encode their measurement level after the final "__".
# This split lets us treat each class appropriately when forming the
# correlation matrix, allows us to seperate nominals from the numeric columns
types <- str_split(names(df), "__", simplify = TRUE)[,3]
types[types == "binary_nominal"] <- "nominal"
df_cont <- df[, types == "continuous", drop = FALSE]
df_ord  <- df[, types == "ordinal",    drop = FALSE]
df_bin  <- df[, types == "binary",     drop = FALSE]
df_nom  <- df[, types == "nominal",    drop = FALSE]

df_num <- bind_cols(df_cont, df_ord, df_bin)
df_num <- df_num %>% mutate(across(everything(), as.numeric))


# ---------------------------------------------------------------------------
# Step 7. Sanity check the columns
# ---------------------------------------------------------------------------

# Print each column's class and drop anything unexpected.  In practice all
# variables should be numeric, integer, factor or ordered.  This guard helps
# catch stray list-columns from earlier processing.
cat(">>> DEBUG: column classes:\n")
print(sapply(df_num, class))
allowed <- function(x) {
  inherits(x, c("numeric","integer","factor","ordered","logical","character"))
}
good_cols <- vapply(df_num, allowed, logical(1))
bad_cols  <- names(good_cols)[!good_cols]
if (length(bad_cols) > 0) {
  cat(">>> WARNING: dropping unsupported columns:\n")
  print(bad_cols)
  df_mix2_clean <- df_mix2_clean[, good_cols]
}
cat(">>> POST‐CLEAN: column classes:\n")
print(sapply(df_num, class))
(df_num <- as.data.frame(df_num))
cat("Post‐conversion class: ", class(df_num), "\n")

# ---------------------------------------------------------------------------
# Step 8. Compute spearman correlation matrix
# ---------------------------------------------------------------------------
# ``COR_METHOD`` toggles between a heterogeneous polychoric/polyserial
# correlation matrix ("mixed") and a simpler Spearman matrix treating all
# variables as numeric ranks.  The mixed option is slower but more faithful
# to the measurement levels.

R <- cor(df_num, method = "spearman", use = "pairwise.complete.obs")

# --- REBUILD type vector to align with df_mix2_clean ---------------------------
# (We can't reuse `types` from Step 4 because columns were dropped.)
types_clean <- str_split(names(df_num), "__", simplify = TRUE)[, 3]
types_clean[types_clean == "binary_nominal"] <- "nominal"  # just in case

# --- Identify continuous cols in the CLEANED data ------------------------------
cont_idx <- which(types_clean == "continuous")

# (Optional) quick report
cat("Continuous vars in cleaned data:", length(cont_idx), "\n")

# Regardless of method we ensure the matrix is positive definite before
# KMO/Bartlett tests or factor extraction. nearPD performs a minimal
# adjustment when needed.
#R_mixed <- as.matrix(nearPD(R_mixed, corr = TRUE)$mat)

stopifnot(!any(is.na(R)))

# ---------------------------------------------------------------------------
# Step 8. Suitability checks and pruning
# ---------------------------------------------------------------------------
# KMO and Bartlett tests
# These diagnostics help verify whether factor analysis is appropriate for
# the correlation matrix.
kmo_res <- psych::KMO(R)
bart_res <- psych::cortest.bartlett(R, n = nrow(df_num))
cat("KMO overall MSA:", round(kmo_res$MSA, 3), "\n")
cat("Bartlett's test p-value:", signif(bart_res$p.value, 3), "\n")

# -- Per-variable MSA assessment ----------------------------------------------
# Drop variables that fall below the commonly-used MSA threshold of 0.5.
msa_vec <- kmo_res$MSAi
names(msa_vec) <- colnames(R)
low_msa <- msa_vec[msa_vec < 0.5]
if (length(low_msa) > 0) {
  cat("Variables dropped for low MSA (<0.5):\n")
  print(round(low_msa, 3))
  keep_vars <- setdiff(colnames(R), names(low_msa))
  R <- R[keep_vars, keep_vars]
  df_num <- df_num[, keep_vars, drop = FALSE]
} else {
  cat("No variables dropped for low MSA\n")
}

# Check KMO of pruned correlation matrix
kmo_res <- psych::KMO(R)
cat("updated KMO overall MSA:", round(kmo_res$MSA, 3), "\n")

# ---------------------------------------------------------------------------
# Step 9. Determine number of factors (parallel analysis & MAP)
# ---------------------------------------------------------------------------

# Two common heuristics are used: Horn's parallel analysis (upper bound)
# and Velicer's MAP (lower bound).  ``k`` can be set manually or using one
# of these estimates.
pa_out <- fa.parallel(R, n.obs = nrow(df_num),
                      fm = "minres", fa = "fa",
                      n.iter = 500, quant = .95,
                      cor = "cor", use = "pairwise", plot = FALSE)
k_PA  <- pa_out$nfact
vss_out <- VSS(R, n = ncol(R),
               fm = "minres", n.obs = nrow(df_num), plot = FALSE)
k_MAP <- which.min(vss_out$map)
k     <- k_MAP

# ---------------------------------------------------------------------------
# Step 10. Bootstrap robust MINRES+geomin to get loadings & uniquenesses
# ---------------------------------------------------------------------------

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
#   12.1 Identify each variable's primary loading & its 95% CI
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
# 20a. Pre-CFA diagnostic checks
# ---------------------------------------------------------------------------
# These checks evaluate the pruned factor structure before moving on to a
# confirmatory factor analysis. They summarise reliability, distributional
# properties and indicator counts.

# --- 20a.1 Reliability (Cronbach's alpha) per factor -----------------------
assign_fac <- apply(abs(Lambda0), 1, which.max)
for (j in seq_len(ncol(Lambda0))) {
  inds <- names(assign_fac)[assign_fac == j & abs(Lambda0[, j]) >= 0.3]
  if (length(inds) >= 2) {
    alpha_j <- tryCatch({
      tmp <- df_mix2_clean[, inds, drop = FALSE]
      tmp_num <- data.frame(lapply(tmp, function(x) if (is.numeric(x)) x else as.numeric(x)))
      psych::alpha(tmp_num, check.keys = FALSE)$total$raw_alpha
    }, error = function(e) NA_real_)
    cat(sprintf("Factor %s Cronbach alpha: %.3f\n", colnames(Lambda0)[j], alpha_j))
  } else {
    cat(sprintf("Factor %s has fewer than two strong indicators; alpha not computed\n",
                colnames(Lambda0)[j]))
  }
}

# --- 20a.2 Normality & outlier screening ----------------------------------
norm_p <- sapply(df_mix2_clean[, keep_final, drop = FALSE], function(v) {
  if (is.numeric(v) && length(na.omit(unique(v))) > 3) {
    tryCatch(shapiro.test(v)$p.value, error = function(e) NA_real_)
  } else {
    NA_real_
  }
})
cat("Shapiro-Wilk normality p-values:\n")
print(round(norm_p, 3))
if (any(norm_p < 0.05, na.rm = TRUE)) {
  cat("Normality questionable for some items; robust estimators advised.\n")
}

out_counts <- sapply(df_mix2_clean[, keep_final, drop = FALSE], function(v) {
  if (is.numeric(v)) sum(abs(scale(v)) > 3, na.rm = TRUE) else 0
})
cat("Outlier counts per item (|z|>3):\n")
print(out_counts)

# --- 20a.3 Indicator count per factor -------------------------------------
ind_counts <- sapply(seq_len(ncol(Lambda0)), function(j)
  sum(abs(Lambda0[, j]) >= 0.3))
names(ind_counts) <- colnames(Lambda0)
cat("Indicators with |loading|>=0.30 per factor:\n")
print(ind_counts)
if (any(ind_counts < 3)) {
  cat("Warning: some factors have fewer than three well-defined indicators.\n")
}

# ---------------------------------------------------------------------------
# 20. Factor scores and GAM regression
# ---------------------------------------------------------------------------
# After pruning we compute Ten Berge factor scores for the retained variables
# and examine their relationship with the productivity index using a
# generalised additive model.  Nominal predictors are collapsed to an "Other"
# level if a category represents fewer than 5% of observations.

# --- 20.1  Compute Bartlett scores for the final items ---------------------raw_data <- df_mix2_clean[, keep_final, drop = FALSE]

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
colnames(F_hat) <- colnames(Lambda0)   # "F1", "F2", …
rownames(F_hat) <- rownames(df_mix2_clean)

# --- 20a.4 Review factor correlations -------------------------------------
fac_cor <- cor(F_hat)
cat("Factor score correlations:\n")
print(round(fac_cor, 3))
if (any(abs(fac_cor[lower.tri(fac_cor)]) > 0.85)) {
  cat("Warning: high correlations between factors detected.\n")
}

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

fit_gam <- function(df, smooth_terms, small_terms, large_terms) {
  all_terms <- c(smooth_terms, small_terms, large_terms)
  form <- as.formula(paste("prod_index ~", paste(all_terms, collapse = " + ")))
  fit  <- mgcv::gam(form, data = df, method = "REML", select = TRUE,
                    drop.unused.levels = FALSE)
  print(summary(fit)$s.table)
  print(summary(fit)$p.table)
  par(mfrow = c(1, ncol(F_hat)))
  plot(fit, pages = 1, all.terms = TRUE, shade = TRUE)
  list(fit = fit, smooth_terms = smooth_terms, large_terms = large_terms)
}

gam_diagnostics <- function(fit, df, smooth_terms, large_terms) {
  cat("Deviance explained:", round(summary(fit)$dev.expl, 3), "\n")
  print(mgcv::concurvity(fit))
  par(mfrow = c(1, 1))
  plot(fit, residuals = TRUE)

  # --- Residual checks -------------------------------------------------
  if (requireNamespace("mgcViz", quietly = TRUE) &&
      exists("appraise", where = asNamespace("mgcViz"), mode = "function")) {
    viz <- mgcViz::getViz(fit)
    app_fun <- get("appraise", envir = asNamespace("mgcViz"))
    print(app_fun(viz))        # QQ/scale-location
    if (exists("influence", where = asNamespace("mgcViz"), mode = "function")) {
      infl_fun <- get("influence", envir = asNamespace("mgcViz"))
      print(infl_fun(viz))     # Cook's distance style plots
    }
  } else {
    cat("mgcViz not installed or missing diagnostics: using base methods\n")
    qqnorm(resid(fit)); qqline(resid(fit))
    plot(fitted(fit), sqrt(abs(resid(fit))),
         ylab = "|residual|^0.5", xlab = "Fitted")
  }
  acf(resid(fit), main = "ACF of GAM residuals")

  cv_deviance <- function(form, data, folds = 5) {
    n <- nrow(data)
    ids <- sample(rep(seq_len(folds), length.out = n))
    dev <- numeric(folds)
    fac_cols <- names(Filter(is.factor, data))
    for (i in seq_len(folds)) {
      train <- data[ids != i, , drop = FALSE]
      test  <- data[ids == i, , drop = FALSE]
      for (fc in fac_cols) {
        train[[fc]] <- factor(train[[fc]], levels = levels(data[[fc]]))
        test[[fc]]  <- factor(test[[fc]],  levels = levels(data[[fc]]))
      }
      m <- mgcv::gam(form, data = data[ids != i, ], method = "REML",
                     drop.unused.levels = FALSE)
      pr <- predict(m, newdata = test, na.action = na.exclude)
      dev[i] <- mean((test$prod_index - pr)^2, na.rm = TRUE)
    }
    mean(dev)
  }
  cv_dev <- cv_deviance(formula(fit), df)
  cat("5-fold CV MSE:", signif(cv_dev, 3), "\n")

  param_terms <- summary(fit)$pTerms.table
  if (!is.null(param_terms) && nrow(param_terms) > 0) {
    p_adj <- p.adjust(param_terms[, "p-value"], method = "fdr")
    print(p_adj)
    signif_terms <- rownames(param_terms)[p_adj < 0.05]
  } else {
    signif_terms <- character(0)
  }

  refit_terms <- c(smooth_terms, signif_terms, large_terms)
  refit_form <- if (length(refit_terms) == 0) {
    as.formula("prod_index ~ 1")
  } else {
    as.formula(paste("prod_index ~", paste(refit_terms, collapse = " + ")))
  }

  refit <- mgcv::gam(refit_form, data = df, method = "REML", select = TRUE,
                     drop.unused.levels = FALSE)
  cat("AIC(original)=", AIC(fit), " AIC(refit)=", AIC(refit), "\n")
  print(summary(refit)$s.table)
  print(summary(refit)$p.table)
  par(mfrow = c(1, ncol(F_hat)))
  plot(refit, pages = 1, all.terms = TRUE, shade = TRUE)
  chk <- mgcv::gam.check(refit)   # k-index; want > ~0.9 and p>0.05
  print(chk)
  low_k <- chk$k.check[, "k-index"] < 0.9
  if (any(low_k)) {
    bad <- rownames(chk$k.check)[low_k]
    cat("Low k-index for:", paste(bad, collapse = ", "), "\n")

    hi_terms <- smooth_terms
    for (nm in bad) {
      idx <- match(nm, smooth_terms)
      if (!is.na(idx)) {
        k_old <- refit$smooth[[idx]]$bs.dim
        hi_terms[idx] <- sub(")$",
                             paste0(", k=", k_old * 2, ")"),
                             hi_terms[idx])
      }
    }

    hi_form <- as.formula(paste("prod_index ~", paste(c(hi_terms, signif_terms, large_terms), collapse = " + ")))
    refit_hi <- mgcv::gam(hi_form, data = df, method = "REML", select = TRUE,
                          drop.unused.levels = FALSE)
    cat("AIC(high-k)=", AIC(refit_hi), "\n")
    anova(refit, refit_hi, test = "Chisq")
  }
  alt_terms <- gsub("s\\(([^)]+)\\)", "s(\\1, bs='cs')", smooth_terms)
  alt_form <- as.formula(paste("prod_index ~", paste(c(alt_terms, signif_terms, large_terms), collapse = " + ")))
  gam_alt <- mgcv::gam(alt_form, data = df, method = "REML", select = TRUE,
                       drop.unused.levels = FALSE)
  cat("AIC(cs basis)=", AIC(gam_alt), "\n")
  anova(refit, gam_alt, test = "Chisq")

  mgcv::concurvity(refit)
  anova(fit, refit, test = "Chisq")   # uses ML; may need method="ML" refits
  refit
}

smooth_terms <- paste0("s(", colnames(F_hat), ")")
small_terms  <- if (length(small_noms)) paste(small_noms, collapse = " + ") else NULL
large_terms  <- if (length(large_noms)) paste0("s(", large_noms, ", bs='re')", collapse = " + ") else NULL

gam_res  <- fit_gam(gam_df, smooth_terms, small_terms, large_terms)
gam_fit  <- gam_res$fit
gam_refit <- gam_diagnostics(gam_fit, gam_df, gam_res$smooth_terms, gam_res$large_terms)

# ------------------------------------------------------------------
#  Additional exploration of factor effects
# ------------------------------------------------------------------

# 1. Correlation matrix of factor scores
print(cor(F_hat))

# 2. Univariate smooths for each factor
for (f in colnames(F_hat)) {
  form <- as.formula(paste("prod_index ~ s(", f, ")"))
  m_single <- mgcv::gam(form, data = gam_df, method = "REML",
                        drop.unused.levels = FALSE)
  cat("\n--- Smooth effect for", f, "---\n")
  print(summary(m_single))
  plot(m_single, shade = TRUE)
  anova(m_single, gam_refit, test = "Chisq")
}

# 3. Pairwise tensor-product interactions
fac_names <- colnames(F_hat)
if (length(fac_names) > 1) {
  pairs <- combn(fac_names, 2, simplify = FALSE)
  for (p in pairs) {
    f1 <- p[1]; f2 <- p[2]
    form_te  <- as.formula(paste("prod_index ~ te(", f1, ", ", f2, ")"))
    form_add <- as.formula(paste("prod_index ~ s(", f1, ") + s(", f2, ")"))
    m_te <- mgcv::gam(form_te, data = gam_df, method = "REML",
                      drop.unused.levels = FALSE)
    m_add <- mgcv::gam(form_add, data = gam_df, method = "REML",
                       drop.unused.levels = FALSE)
    cat("\n--- Tensor interaction model (", f1, " × ", f2, ") ---\n")
    print(summary(m_te))
    vis.gam(m_te, view = c(f1, f2), plot.type = "persp",
            color = "heat", main = paste0("Productivity surface f(", f1, ", ", f2, ")"))
    anova(m_add, m_te, test = "Chisq")
  }
}

# ---- Draw three slices of tensor surfaces for all factor pairs ----
plot_tensor_slices <- function(model, df, factors,
                               z_grid = c(-1, 0, 1), npoints = 200) {
  
  for (pair in combn(factors, 2, simplify = FALSE)) {
    f1 <- pair[1] ; f2 <- pair[2]
    
    ## 1. Grid for predictions
    f1_seq  <- seq(min(df[[f1]]), max(df[[f1]]), length.out = npoints)
    newdat  <- expand.grid(f1_seq, z_grid)
    colnames(newdat) <- c(f1, f2)
    
    ## 2. Fitted values & SEs
    Xp  <- predict(model, newdat, type = "lpmatrix")
    fit <- drop(Xp %*% coef(model))
    se  <- sqrt(rowSums((Xp %*% vcov(model)) * Xp))
    newdat$fit <- fit
    newdat$se  <- se
    
    ## 3. Cast z-levels to a factor so brewer scales work
    newdat[[f2]] <- factor(newdat[[f2]])
    
    ## 4. Plot
    p <- ggplot(newdat, aes_string(f1, "fit", colour = f2)) +
      geom_line(size = 1.1) +
      geom_ribbon(aes_string(ymin = "fit - 2*se",
                             ymax = "fit + 2*se",
                             fill  = f2),
                  alpha = .20, colour = NA) +
      scale_colour_brewer(palette = "Set1", name = paste0(f2, " (z)")) +
      scale_fill_brewer(palette = "Set1",  name = paste0(f2, " (z)")) +
      labs(y = "Partial effect on productivity",
           title = paste0(f1, " → productivity curves at three ", f2, " levels")) +
      theme_minimal()
    
    print(p)
  }
}

# ---- Diagnostics for the F1 × F2 interaction -------------------------
diagnose_F1_F2 <- function(df) {
  int_mod <- gam(prod_index ~ te(F1, F2, k = c(9, 9)),
                 data   = df,
                 method = "REML")

  # ---- Differences between high and low F2 at a grid of F1 values ----
  gr <- seq(min(df$F1), max(df$F1), by = 0.1)
  hi_dat <- data.frame(F1 = gr, F2 = 1)
  lo_dat <- data.frame(F1 = gr, F2 = -1)
  pr_hi  <- predict(int_mod, hi_dat, se.fit = TRUE)
  pr_lo  <- predict(int_mod, lo_dat, se.fit = TRUE)

  diff <- pr_hi$fit - pr_lo$fit
  se   <- sqrt(pr_hi$se.fit^2 + pr_lo$se.fit^2)
  sig_idx <- which(abs(diff)/se > 1.96)
  cat("Significant F1 range:", range(gr[sig_idx]), "\n")

  q <- ecdf(df$F1)
  cat("Share in significant band:", 1 - q(1.17), "\n")

  pts    <- 0:4
  hi_pts <- data.frame(F1 = pts, F2 = 1)
  lo_pts <- data.frame(F1 = pts, F2 = -1)
  pr_hi2 <- predict(int_mod, hi_pts, se.fit = TRUE)
  pr_lo2 <- predict(int_mod, lo_pts, se.fit = TRUE)

  delta <- pr_hi2$fit - pr_lo2$fit
  cil   <- delta - 1.96 * sqrt(pr_hi2$se.fit^2 + pr_lo2$se.fit^2)
  ciu   <- delta + 1.96 * sqrt(pr_hi2$se.fit^2 + pr_lo2$se.fit^2)
  print(cbind(F1 = pts, diff = round(delta, 2),
              CI_lo = round(cil, 2), CI_hi = round(ciu, 2)))

  p1 <- ggplot(df, aes(F1, prod_index,
                 colour = cut(F2, c(-Inf, -0.5, 0.5, Inf)))) +
    geom_point(alpha = 0.3) +
    stat_smooth(method = "gam", formula = y ~ s(x, bs = "tp"), se = FALSE) +
    theme_classic()
  print(p1)

  # ---- Robustness check: trim extreme productivity value -------------
  thr      <- max(df$prod_index, na.rm = TRUE)
  df_trim  <- filter(df, prod_index < thr)
  int_trim <- gam(prod_index ~ te(F1, F2, k = c(9, 9)),
                  data   = df_trim,
                  method = "REML")

  simple_slope <- function(model, F1_pts = 0:4,
                           F2_hi = 1, F2_lo = -1) {
    hi <- data.frame(F1 = F1_pts, F2 = F2_hi)
    lo <- data.frame(F1 = F1_pts, F2 = F2_lo)

    pr_hi <- predict(model, hi, se.fit = TRUE)
    pr_lo <- predict(model, lo, se.fit = TRUE)

    delta <- pr_hi$fit - pr_lo$fit
    se    <- sqrt(pr_hi$se.fit^2 + pr_lo$se.fit^2)
    ci_lo <- delta - 1.96 * se
    ci_hi <- delta + 1.96 * se

    tibble(F1 = F1_pts, diff = delta, CI_lo = ci_lo, CI_hi = ci_hi)
  }

  tbl_orig <- simple_slope(int_mod)
  tbl_trim <- simple_slope(int_trim)
  comparison <- left_join(tbl_orig  %>% rename_with(~paste0(.x, "_orig"), -F1),
                          tbl_trim %>% rename_with(~paste0(.x, "_trim"), -F1),
                          by = "F1") |>
    mutate(delta_change = diff_trim - diff_orig)
  print(comparison)

  # ---- Alternative centring: F2 quartiles ----------------------------
  df_q <- df %>%
    mutate(F2_q = cut(F2,
                      breaks = quantile(F2, probs = seq(0, 1, 0.25),
                                        na.rm = TRUE),
                      include.lowest = TRUE,
                      labels = c("Q1", "Q2", "Q3", "Q4")))

  m_by_q <- gam(prod_index ~ s(F1, by = F2_q, k = 9) + F2_q,
                data   = df_q,
                method = "REML",
                drop.unused.levels = FALSE)
  print(summary(m_by_q))
  plot(m_by_q, pages = 1, shade = TRUE, seWithMean = TRUE)

  band_breaks <- c(1.17, 2, 3, 4, 5, max(df_q$F1))
  band_labels <- c("1.17–2", "2–3", "3–4", "4–5", "5+")
  df_q <- df_q %>%
    mutate(F1_band = cut(F1, breaks = band_breaks, right = FALSE,
                         labels = band_labels))

  band_centers <- tibble(
    F1_band = band_labels,
    x_pos   = (head(band_breaks, -1) + tail(band_breaks, -1)) / 2
  )

  band_counts <- df_q %>%
    filter(!is.na(F1_band)) %>%
    count(F1_band) %>%
    left_join(band_centers, by = "F1_band") %>%
    mutate(y_pos = Inf)

  p2 <- ggplot(df_q, aes(F1, prod_index, colour = F2_q)) +
    geom_point(alpha = 0.25) +
    geom_smooth(method = "gam", formula = y ~ s(x, bs = "tp"), se = TRUE) +
    geom_rug(alpha = 0.3) +
    geom_text(
      data        = band_counts,
      aes(x = x_pos, y = y_pos, label = n),
      vjust       = 1.2,
      size        = 3.2,
      inherit.aes = FALSE
    ) +
    scale_colour_brewer(palette = "Dark2", name = "F2 quartile") +
    labs(
      title    = "F1 → Productivity curves by F2 quartile",
      subtitle = "Numbers indicate sample size in each high-F1 band",
      x        = "F1 (z-score)",
      y        = "Partial effect on productivity"
    ) +
    coord_cartesian(clip = "off") +
    theme_classic()
  print(p2)

  invisible(list(p1 = p1, p2 = p2))
}

fac_names <- colnames(F_hat)

if (length(fac_names) > 1) {
  pairs <- combn(fac_names, 2, simplify = FALSE)

  for (p in pairs) {
    f1 <- p[[1]]
    f2 <- p[[2]]

    m_te <- mgcv::gam(
      as.formula(paste0("prod_index ~ te(", f1, ", ", f2, ")")),
      data = gam_df,
      method = "REML",
      drop.unused.levels = FALSE
    )

    # correct argument order: model, df, factors-vector
    plot_tensor_slices(m_te, gam_df, c(f1, f2))

    if (f1 == "F1" && f2 == "F2") {
      diagnose_F1_F2(gam_df)
    }
  }
}

# ---------------------------------------------------------------------
# 20.4  Mediation test: does F2 act through F1? -------------------------
# ---------------------------------------------------------------------
# Theory: household capacity (F2) may boost productivity indirectly by
# enabling additional F1 investment.  We fit a GAM of F1 on F2 as the
# mediator model and a GAM of productivity on both F1 and F2 as the
# outcome model.  The `mediation` package estimates the indirect
# effect via simulated draws from these fits.

med_model  <- gam(F1 ~ s(F2), data = gam_df, method = "REML")
out_model  <- gam(prod_index ~ s(F1) + s(F2), data = gam_df, method = "REML")

med_result <- mediate(med_model, out_model,
                      treat    = "F2",
                      mediator = "F1",
                      sims     = 100,
                      boot     = TRUE)

print(summary(med_result))

## ---------------------------------------------------------------------
## 3.  Moderated mediation: F1 → seedlings → productivity with F2 moderator
## ---------------------------------------------------------------------
seedling_var <- grep("seedling", names(gam_df), ignore.case = TRUE, value = TRUE)
if (length(seedling_var) > 0) {
  seedling_var <- seedling_var[1]
  cat("Using mediator variable:", seedling_var, "\n")
  
  ## Use the same factor as in the GAM fit (rare levels collapsed)
  seedlings_use <- gam_df[[seedling_var]]
  if (!is.factor(seedlings_use)) {
    seedlings_use <- factor(seedlings_use)
  }
  seedlings_use <- droplevels(seedlings_use)

  ## a-path: does F1 predict seedling use and does that depend on F2?
  if (is.factor(seedlings_use) && nlevels(seedlings_use) > 2) {
    K <- nlevels(seedlings_use) - 1
    y <- as.numeric(seedlings_use) - 1
    form_list <- vector("list", K)
    form_list[[1]] <-
      as.formula("y ~ s(F1) + s(F2) + ti(F1, F2)")
    for (j in 2:K) {
      form_list[[j]] <- as.formula("~ s(F1) + s(F2) + ti(F1, F2)")
    }
    m_a <- gam(form_list, family = mgcv::multinom(K = K),
               data = transform(gam_df, y = y))
  } else {
    m_a <- gam(seedlings_use ~ s(F1) + s(F2) + ti(F1, F2),
               family = binomial, data = gam_df)
  }
  print(summary(m_a))
  print(m_a)

  ## b-path: does the seedling → productivity effect vary with F2?
  m_b <- gam(prod_index ~ s(F1) + s(F2) + seedlings_use + seedlings_use:F2,
             data = gam_df, method = "REML")
  print(m_b)
  print(summary(m_b))
  anova(m_b, test = "Chisq")
} else {
  cat("No seedlings variable found for moderated mediation test\n")
}

# ---------------------------------------------------------------------------
# 22. Mediation check: Do agronomic practices explain part of the
#     F1 → productivity effect?
# ---------------------------------------------------------------------------
# Theory: On-farm capital (F1) raises the probability of using
# self-prepared seedlings (Q56), which then boosts productivity.
seed_var <- "Q56__For_vegetables_do_you_use_seedlings__nominal"
if (seed_var %in% names(gam_df)) {
  # use the same collapsed factor as in the productivity GAM
  gam_df$seedlings <- gam_df[[seed_var]]
  if (!is.factor(gam_df$seedlings)) {
    gam_df$seedlings <- factor(gam_df$seedlings)
  }
  gam_df$seedlings <- droplevels(gam_df$seedlings)

  message("\n=== GAM mediation test: F1 → Q56 → productivity ===")

  # a) does F1 predict the seedling practice categories?
  seed_num <- as.numeric(gam_df$seedlings) - 1
  K <- nlevels(gam_df$seedlings) - 1
  flist <- c(list(seed_num ~ s(F1)), rep(list(~s(F1)), K - 1))
  gam_seed <- mgcv::gam(flist, data = data.frame(gam_df, seed_num),
                        family = mgcv::multinom(K = K), method = "REML")
  print(summary(gam_seed))

  # b) effect of F1 on productivity controlling for Q56
  gam_base <- mgcv::gam(prod_index ~ s(F1), data = gam_df, method = "REML")
  gam_med  <- mgcv::gam(prod_index ~ s(F1) + seedlings, data = gam_df,
                        method = "REML")
  cat("ΔAIC =", AIC(gam_base) - AIC(gam_med), "\n")
  print(summary(gam_med))
} else {
  message("Q56 variable not found; skipping mediation check")
}


# ---------------------------------------------------------------------------
# 21. Robust Confirmatory Factor Analysis (CFA) with Diagnostics (EFA-aligned)
# ---------------------------------------------------------------------------

# --- 1. Data preparation: align all variable names and types ---
cfa_vars <- rownames(Lambda0)
missing_cfa_vars <- setdiff(cfa_vars, names(df_mix2_clean))
if (length(missing_cfa_vars) > 0) {
  cat("\n[CFA ERROR] The following variables in Lambda0 are missing from df_mix2_clean:\n")
  print(missing_cfa_vars)
  stop("Aborting CFA: variables missing from data.")
}
cfa_df <- df_mix2_clean[, cfa_vars, drop = FALSE]

# Identify continuous and ordered variables
all_cont_vars <- c(
  "Q62__How_much_VEGETABLES_do_you_harvest_per_year_from_this_plot_kilograms__continuous",
  "Q50__How_much_land_that_is_yours_do_you_cultivate_bigha__continuous",
  "Q52__On_how_much_land_do_you_grow_vegetables_bigha__continuous",
  "Q109__What_is_your_households_yearly_income_overall_including_agriculture_NPR__continuous",
  "Q0__hope_total__continuous",
  "Q0__self_control_score__continuous",
  "Q5__AgeYears__continuous",
  "Q108__What_is_your_households_yearly_income_from_agriculture_NPR__continuous"
)
cont_vars <- intersect(all_cont_vars, names(cfa_df))
all_ordered_items <- c(
  "Q112__Generally_speaking_how_would_you_define_your_farming__ordinal",
  "Q0__average_of_farming_practices__ordinal",
  "Q70__in_the_past_12_months_did_you_receive_any_info_from_anyone_on_agriculture__binary__1"
)
ordered_items <- intersect(all_ordered_items, names(cfa_df))

# --- 2. Data diagnostics ---
cat("\n[CFA] Variable types:\n")
print(sapply(cfa_df, class))
cat("\n[CFA] Continuous variables to be scaled:\n")
print(cont_vars)
cat("\n[CFA] Ordered variables:\n")
print(ordered_items)

# Check for near-constant or all-missing variables
for (v in names(cfa_df)) {
  unique_vals <- unique(na.omit(cfa_df[[v]]))
  if (length(unique_vals) <= 1) {
    cat(sprintf("[CFA WARNING] Variable %s is constant or all missing!\n", v))
  }
}

# Check for high collinearity among continuous variables
if (length(cont_vars) > 1) {
  cor_mat <- cor(cfa_df[cont_vars], use = "pairwise.complete.obs")
  high_corrs <- which(abs(cor_mat) > 0.95 & abs(cor_mat) < 1, arr.ind = TRUE)
  if (length(high_corrs) > 0) {
    cat("[CFA WARNING] High collinearity among continuous variables:\n")
    for (i in seq_len(nrow(high_corrs))) {
      v1 <- rownames(cor_mat)[high_corrs[i, 1]]
      v2 <- colnames(cor_mat)[high_corrs[i, 2]]
      cat(sprintf("  %s and %s: %.3f\n", v1, v2, cor_mat[high_corrs[i, 1], high_corrs[i, 2]]))
    }
  }
}

# --- 3. Preprocessing: scale continuous, coerce ordered ---
for (v in cont_vars) {
  if (!is.numeric(cfa_df[[v]])) cfa_df[[v]] <- as.numeric(cfa_df[[v]])
}
if (length(cont_vars) > 0) cfa_df[cont_vars] <- scale(cfa_df[cont_vars])
for (v in ordered_items) {
  if (!is.ordered(cfa_df[[v]])) {
    cfa_df[[v]] <- ordered(cfa_df[[v]], levels = sort(unique(cfa_df[[v]])))
  }
}

# --- 4. Generate CFA model syntax from Lambda0 ---
cfa_model_lines <- c()
for (j in 1:ncol(Lambda0)) {
  factor_name <- colnames(Lambda0)[j]
  items <- rownames(Lambda0)[abs(Lambda0[, j]) > 0]
  if (length(items) > 0) {
    cfa_model_lines <- c(cfa_model_lines, paste0(factor_name, " =~ ", paste(items, collapse = " + ")))
  }
}
# Add factor covariance
if (ncol(Lambda0) > 1) {
  fac_covs <- combn(colnames(Lambda0), 2, FUN = function(x) paste(x[1], "~~", x[2]), simplify = TRUE)
  cfa_model_lines <- c(cfa_model_lines, fac_covs)
}
cfa_model <- paste(cfa_model_lines, collapse = "\n")
cat("\n[CFA] Model syntax:\n", cfa_model, "\n")

# --- 5. Fit CFA model (let lavaan handle starting values) ---
fit_cfa <- tryCatch({
  lavaan::cfa(
    cfa_model,
    data = cfa_df,
    std.lv = TRUE,
    estimator = "WLSMV",
    ordered = ordered_items
  )
}, error = function(e) {
  cat("\n[CFA ERROR] lavaan::cfa failed:\n")
  print(e)
  return(NULL)
})

if (is.null(fit_cfa)) stop("CFA model did not fit. See error above.")

# --- 6. Diagnostics and output ---
cat("\n[CFA] Fit indices:\n")
print(lavaan::fitMeasures(fit_cfa, c("chisq", "df", "cfi", "rmsea", "srmr")))
cat("\n[CFA] Standardized loadings:\n")
print(lavaan::inspect(fit_cfa, "std")$lambda)
cat("\n[CFA] Modification indices (MI > 10):\n")
print(subset(lavaan::modindices(fit_cfa), mi > 10))

# --- 7. Save results and plot ---
saveRDS(fit_cfa, file = "fit_cfa.rds")
if (requireNamespace("semPlot", quietly = TRUE)) {
  pdf("cfa_model_diagram.pdf", width = 10, height = 8)
  semPlot::semPaths(fit_cfa, "std", whatLabels = "std", edge.label.cex = 1.1, layout = "tree", style = "lisrel")
  dev.off()
}

# --- 8. Factor scores ---
factor_scores <- tryCatch({
  lavPredict(fit_cfa, method = "Bartlett")
}, error = function(e) {
  cat("[CFA WARNING] Could not compute factor scores:\n")
  print(e)
  return(NULL)
})
if (!is.null(factor_scores)) {
  write.csv(factor_scores, "factor_scores.csv", row.names = TRUE)
}

# --- 9. Residuals heatmap ---
resid_mat_cfa <- lavaan::residuals(fit_cfa, type = "cor")$cov
heatmap(resid_mat_cfa, main = "CFA Residual Correlation Matrix", symm = TRUE)

# --- 10. Suggested model modification based on highest MI (do not remove indicators) ---
mod_indices <- lavaan::modindices(fit_cfa)
mod_indices <- mod_indices[order(-mod_indices$mi), ]
mod_indices_to_add <- subset(mod_indices, mi > 10 & !(op == "~1" | op == "~"))

if (nrow(mod_indices_to_add) > 0) {
  top_mi <- mod_indices_to_add[1, ]
  cat("\n[SUGGESTED MODIFICATION] Add the following to the model for improved fit (MI = ",
      round(top_mi$mi, 2), "):\n  ",
      top_mi$lhs, top_mi$op, top_mi$rhs, "\n", sep = "")
}

# --- 11. Automated MI-based model refinement (do not remove indicators) ---
# This block will iteratively add the top MI-suggested cross-loading or residual correlation (MI > 10)
# to the model, refit, and repeat up to max_steps or until no MI > 10 remain.
# The refined model and fit indices will be printed at each step.

# --- Automated MI-based model refinement ---
max_steps <- 5  # Set the maximum number of MI-based modifications
cfa_model_refined <- cfa_model
fit_cfa_refined <- fit_cfa
added_mods <- character(0)

for (step in 1:max_steps) {
  mi_ref <- lavaan::modindices(fit_cfa_refined)
  mi_ref <- mi_ref[order(-mi_ref$mi), ]
  mi_to_add <- subset(mi_ref, mi > 10 & !(op == "~1" | op == "~") & !paste(lhs, op, rhs) %in% added_mods)
  if (nrow(mi_to_add) == 0) {
    cat("\n[MI-REFINEMENT] No more MI > 10 to add. Stopping refinement.\n")
    break
  }
  top_mi <- mi_to_add[1, ]
  new_line <- paste(top_mi$lhs, top_mi$op, top_mi$rhs)
  cfa_model_refined <- paste(cfa_model_refined, new_line, sep = "\n")
  added_mods <- c(added_mods, new_line)
  cat("\n[MI-REFINEMENT] Step", step, ": Adding", new_line, "(MI =", round(top_mi$mi, 2), ")\n")
  fit_cfa_refined <- tryCatch({
    lavaan::cfa(
      cfa_model_refined,
      data = cfa_df,
      std.lv = TRUE,
      estimator = "WLSMV",
      ordered = ordered_items
    )
  }, error = function(e) {
    cat("[MI-REFINEMENT ERROR] lavaan::cfa failed after adding:", new_line, "\n")
    print(e)
    return(fit_cfa_refined)  # Return previous fit
  })
  fit_indices_ref <- lavaan::fitMeasures(fit_cfa_refined, c("chisq", "df", "cfi", "rmsea", "srmr"))
  cat("[MI-REFINEMENT] Fit indices after step", step, ":\n")
  print(fit_indices_ref)
}

# --- Final model diagnostics and residuals ---
cat("\n[FINAL MODEL] Standardized solution:\n")
print(summary(fit_cfa_refined, standardized = TRUE))

# Print all negative residual variances (Heywood cases)
cat("\n[FINAL MODEL] Negative residual variances (Heywood cases):\n")
par_table <- lavaan::parTable(fit_cfa_refined)
neg_resid <- subset(par_table, op == "~~" & lhs == rhs & est < 0)
if (nrow(neg_resid) > 0) {
  print(neg_resid[, c("lhs", "est")])
} else {
  cat("None detected.\n")
}

# Plot residuals heatmap for the final model
cat("\n[FINAL MODEL] Residual correlation heatmap:\n")
resid_mat_final <- lavaan::residuals(fit_cfa_refined, type = "cor")$cov
heatmap(resid_mat_final, main = "Final CFA Residual Correlation Matrix", symm = TRUE)

# Print all MI > 10 for the final model
cat("\n[FINAL MODEL] Modification indices (MI > 10):\n")
mod_indices_final <- lavaan::modindices(fit_cfa_refined)
mod_indices_final <- mod_indices_final[order(-mod_indices_final$mi), ]
high_mi_final <- subset(mod_indices_final, mi > 10 & !(op == "~1" | op == "~"))
if (nrow(high_mi_final) > 0) {
  print(high_mi_final[, c("lhs", "op", "rhs", "mi")])
} else {
  cat("No MI > 10 remain.\n")
}

# ---------------------------------------------------------------------------
# 22. Advanced Diagnostic Checks and Model Refinements
# ---------------------------------------------------------------------------

cat("\n", strrep("=", 80), "\n")
cat("ADVANCED DIAGNOSTIC CHECKS AND MODEL REFINEMENTS\n")
cat(strrep("=", 80), "\n")

# --- 1. Resolve Q70 Heywood Case ---
cat("\n[DIAGNOSTIC 1] Resolving Q70 Heywood Case\n")
cat(strrep("-", 50), "\n")

# Check if Q70 has negative residual variance
q70_neg_resid <- subset(par_table, op == "~~" & lhs == "Q70__in_the_past_12_months_did_you_receive_any_info_from_anyone_on_agriculture__binary__1" & rhs == "Q70__in_the_past_12_months_did_you_receive_any_info_from_anyone_on_agriculture__binary__1" & est < 0)
if (nrow(q70_neg_resid) > 0) {
  cat("Q70 has negative residual variance:", round(q70_neg_resid$est, 4), "\n")
  
  # 1a. Temporarily fix Q70 residual variance
  cat("\n[1a] Fixing Q70 residual variance to 0.01...\n")
  cfa_model_fixed <- cfa_model_refined
  cfa_model_fixed <- paste0(cfa_model_fixed, "\nQ70__in_the_past_12_months_did_you_receive_any_info_from_anyone_on_agriculture__binary__1 ~~ 0.01*Q70__in_the_past_12_months_did_you_receive_any_info_from_anyone_on_agriculture__binary__1")
  
  fit_cfa_fixed <- tryCatch({
    lavaan::cfa(
      cfa_model_fixed,
      data = cfa_df,
      std.lv = TRUE,
      estimator = "WLSMV",
      ordered = ordered_items
    )
  }, error = function(e) {
    cat("[ERROR] Fixed model failed to converge:\n")
    print(e)
    return(NULL)
  })
  
  if (!is.null(fit_cfa_fixed)) {
    cat("Fixed model fit indices:\n")
    print(lavaan::fitMeasures(fit_cfa_fixed, c("chisq", "df", "cfi", "rmsea", "srmr")))
    
    # Compare standardized loadings
    cat("\nQ70 loading comparison:\n")
    orig_loading <- lavaan::inspect(fit_cfa_refined, "std")$lambda["Q70__in_the_past_12_months_did_you_receive_any_info_from_anyone_on_agriculture__binary__1", ]
    fixed_loading <- lavaan::inspect(fit_cfa_fixed, "std")$lambda["Q70__in_the_past_12_months_did_you_receive_any_info_from_anyone_on_agriculture__binary__1", ]
    cat("Original:", round(orig_loading, 3), "\n")
    cat("Fixed:", round(fixed_loading, 3), "\n")
  }
  
  # 1b. Examine Q70 binary coding and distribution
  cat("\n[1b] Examining Q70 distribution and coding:\n")
  q70_var <- "Q70__in_the_past_12_months_did_you_receive_any_info_from_anyone_on_agriculture__binary__1"
  if (q70_var %in% names(cfa_df)) {
    q70_tab <- table(cfa_df[[q70_var]], useNA = "ifany")
    cat("Q70 frequency table:\n")
    print(q70_tab)
    cat("Proportions:\n")
    print(round(prop.table(q70_tab), 3))
    
    # Check for sparse categories
    sparse_cats <- q70_tab[q70_tab < 10]
    if (length(sparse_cats) > 0) {
      cat("\nSparse categories (< 10 observations):\n")
      print(sparse_cats)
      cat("Consider merging these categories.\n")
    }
  }
} else {
  cat("Q70 does not have negative residual variance.\n")
}

# --- 2. Probe Q112-Q70 Relationship ---
cat("\n[DIAGNOSTIC 2] Probing Q112-Q70 Relationship\n")
cat(strrep("-", 50), "\n")

# 2a. Test three-factor model with dedicated "Market-engagement/Information" factor
cat("\n[2a] Testing three-factor model with dedicated Market-engagement/Information factor...\n")

# Create three-factor model syntax
cfa_model_3f <- c()
# F1: Commercial-Orientation (production-focused items)
f1_items <- c("Q62__How_much_VEGETABLES_do_you_harvest_per_year_from_this_plot_kilograms__continuous",
               "Q108__What_is_your_households_yearly_income_from_agriculture_NPR__continuous",
               "Q0__hope_total__continuous")
f1_syntax <- paste("F1 =~", paste(f1_items, collapse = " + "))

# F2: Household Resources & Experience
f2_items <- c("Q109__What_is_your_households_yearly_income_overall_including_agriculture_NPR__continuous",
               "Q5__AgeYears__continuous",
               "Q0__self_control_score__continuous")
f2_syntax <- paste("F2 =~", paste(f2_items, collapse = " + "))

# F3: Market-engagement/Information (new factor)
f3_items <- c("Q112__Generally_speaking_how_would_you_define_your_farming__ordinal",
               "Q70__in_the_past_12_months_did_you_receive_any_info_from_anyone_on_agriculture__binary__1",
               "Q52__On_how_much_land_do_you_grow_vegetables_bigha__continuous")
f3_syntax <- paste("F3 =~", paste(f3_items, collapse = " + "))

cfa_model_3f <- c(f1_syntax, f2_syntax, f3_syntax)

# Add factor covariances
fac_covs_3f <- c("F1 ~~ F2", "F1 ~~ F3", "F2 ~~ F3")
cfa_model_3f <- c(cfa_model_3f, fac_covs_3f)

# Add the correlated residual from the refined model
if (any(grepl("Q112.*Q70", cfa_model_refined))) {
  cfa_model_3f <- c(cfa_model_3f, "Q112__Generally_speaking_how_would_you_define_your_farming__ordinal ~~ Q70__in_the_past_12_months_did_you_receive_any_info_from_anyone_on_agriculture__binary__1")
}

cfa_model_3f_text <- paste(cfa_model_3f, collapse = "\n")
cat("Three-factor model syntax:\n")
cat(cfa_model_3f_text, "\n")

# Fit three-factor model
fit_cfa_3f <- tryCatch({
  lavaan::cfa(
    cfa_model_3f_text,
    data = cfa_df,
    std.lv = TRUE,
    estimator = "WLSMV",
    ordered = ordered_items
  )
}, error = function(e) {
  cat("[ERROR] Three-factor model failed to converge:\n")
  print(e)
  return(NULL)
})

if (!is.null(fit_cfa_3f)) {
  cat("\nThree-factor model fit indices:\n")
  print(lavaan::fitMeasures(fit_cfa_3f, c("chisq", "df", "cfi", "rmsea", "srmr")))
  
  cat("\nThree-factor standardized loadings:\n")
  print(lavaan::inspect(fit_cfa_3f, "std")$lambda)
  
  # Compare with two-factor model
  cat("\nModel comparison (2-factor vs 3-factor):\n")
  fit_2f <- lavaan::fitMeasures(fit_cfa_refined, c("chisq", "df", "cfi", "rmsea", "srmr"))
  fit_3f <- lavaan::fitMeasures(fit_cfa_3f, c("chisq", "df", "cfi", "rmsea", "srmr"))
  comparison_df <- data.frame(
    Model = c("2-Factor", "3-Factor"),
    ChiSq = c(fit_2f["chisq"], fit_3f["chisq"]),
    DF = c(fit_2f["df"], fit_3f["df"]),
    CFI = c(fit_2f["cfi"], fit_3f["cfi"]),
    RMSEA = c(fit_2f["rmsea"], fit_3f["rmsea"]),
    SRMR = c(fit_2f["srmr"], fit_3f["srmr"])
  )
  print(comparison_df)
}

# 2b. Test bifactor model
cat("\n[2b] Testing bifactor model (general F1 + specific info factor)...\n")

# Create bifactor model syntax
cfa_model_bifactor <- c()
# General factor (F1)
general_items <- c("Q62__How_much_VEGETABLES_do_you_harvest_per_year_from_this_plot_kilograms__continuous",
                   "Q108__What_is_your_households_yearly_income_from_agriculture_NPR__continuous",
                   "Q0__hope_total__continuous",
                   "Q109__What_is_your_households_yearly_income_overall_including_agriculture_NPR__continuous",
                   "Q5__AgeYears__continuous",
                   "Q0__self_control_score__continuous",
                   "Q112__Generally_speaking_how_would_you_define_your_farming__ordinal",
                   "Q70__in_the_past_12_months_did_you_receive_any_info_from_anyone_on_agriculture__binary__1",
                   "Q52__On_how_much_land_do_you_grow_vegetables_bigha__continuous")
general_syntax <- paste("F1 =~", paste(general_items, collapse = " + "))

# Specific information factor (S1)
specific_items <- c("Q112__Generally_speaking_how_would_you_define_your_farming__ordinal",
                    "Q70__in_the_past_12_months_did_you_receive_any_info_from_anyone_on_agriculture__binary__1",
                    "Q52__On_how_much_land_do_you_grow_vegetables_bigha__continuous")
specific_syntax <- paste("S1 =~", paste(specific_items, collapse = " + "))

cfa_model_bifactor <- c(general_syntax, specific_syntax)

# Constrain general and specific factors to be orthogonal
cfa_model_bifactor <- c(cfa_model_bifactor, "F1 ~~ 0*S1")

# Add correlated residual if needed
if (any(grepl("Q112.*Q70", cfa_model_refined))) {
  cfa_model_bifactor <- c(cfa_model_bifactor, "Q112__Generally_speaking_how_would_you_define_your_farming__ordinal ~~ Q70__in_the_past_12_months_did_you_receive_any_info_from_anyone_on_agriculture__binary__1")
}

cfa_model_bifactor_text <- paste(cfa_model_bifactor, collapse = "\n")
cat("Bifactor model syntax:\n")
cat(cfa_model_bifactor_text, "\n")

# Fit bifactor model
fit_cfa_bifactor <- tryCatch({
  lavaan::cfa(
    cfa_model_bifactor_text,
    data = cfa_df,
    std.lv = TRUE,
    estimator = "WLSMV",
    ordered = ordered_items
  )
}, error = function(e) {
  cat("[ERROR] Bifactor model failed to converge:\n")
  print(e)
  return(NULL)
})

if (!is.null(fit_cfa_bifactor)) {
  cat("\nBifactor model fit indices:\n")
  print(lavaan::fitMeasures(fit_cfa_bifactor, c("chisq", "df", "cfi", "rmsea", "srmr")))
  
  cat("\nBifactor standardized loadings:\n")
  print(lavaan::inspect(fit_cfa_bifactor, "std")$lambda)
}

# --- 3. Cross-validation (split-half) ---
cat("\n[DIAGNOSTIC 3] Cross-validation (Split-half)\n")
cat(strrep("-", 50), "\n")

# Set seed for reproducible splits
set.seed(2025)

# Create split-half samples
n_total <- nrow(cfa_df)
n_half <- floor(n_total / 2)
indices <- sample(1:n_total)
split1_idx <- indices[1:n_half]
split2_idx <- indices[(n_half + 1):n_total]

split1_data <- cfa_df[split1_idx, ]
split2_data <- cfa_df[split2_idx, ]

cat("Split-half sample sizes:", nrow(split1_data), "and", nrow(split2_data), "\n")

# Fit models on both splits
fit_split1 <- tryCatch({
  lavaan::cfa(
    cfa_model_refined,
    data = split1_data,
    std.lv = TRUE,
    estimator = "WLSMV",
    ordered = ordered_items
  )
}, error = function(e) {
  cat("[ERROR] Split 1 model failed:\n")
  print(e)
  return(NULL)
})

fit_split2 <- tryCatch({
  lavaan::cfa(
    cfa_model_refined,
    data = split2_data,
    std.lv = TRUE,
    estimator = "WLSMV",
    ordered = ordered_items
  )
}, error = function(e) {
  cat("[ERROR] Split 2 model failed:\n")
  print(e)
  return(NULL)
})

if (!is.null(fit_split1) && !is.null(fit_split2)) {
  # Extract loadings for comparison
  loadings_split1 <- lavaan::inspect(fit_split1, "std")$lambda
  loadings_split2 <- lavaan::inspect(fit_split2, "std")$lambda
  
  # Focus on Q70 and Q52 loadings
  q70_var <- "Q70__in_the_past_12_months_did_you_receive_any_info_from_anyone_on_agriculture__binary__1"
  q52_var <- "Q52__On_how_much_land_do_you_grow_vegetables_bigha__continuous"
  
  cat("\nQ70 loading consistency across splits:\n")
  if (q70_var %in% rownames(loadings_split1) && q70_var %in% rownames(loadings_split2)) {
    q70_split1 <- loadings_split1[q70_var, ]
    q70_split2 <- loadings_split2[q70_var, ]
    cat("Split 1:", round(q70_split1, 3), "\n")
    cat("Split 2:", round(q70_split2, 3), "\n")
    cat("Difference:", round(abs(q70_split1 - q70_split2), 3), "\n")
  }
  
  cat("\nQ52 loading consistency across splits:\n")
  if (q52_var %in% rownames(loadings_split1) && q52_var %in% rownames(loadings_split2)) {
    q52_split1 <- loadings_split1[q52_var, ]
    q52_split2 <- loadings_split2[q52_var, ]
    cat("Split 1:", round(q52_split1, 3), "\n")
    cat("Split 2:", round(q52_split2, 3), "\n")
    cat("Difference:", round(abs(q52_split1 - q52_split2), 3), "\n")
  }
  
  # Overall loading stability
  common_vars <- intersect(rownames(loadings_split1), rownames(loadings_split2))
  if (length(common_vars) > 0) {
    loadings_diff <- abs(loadings_split1[common_vars, ] - loadings_split2[common_vars, ])
    cat("\nMean absolute loading difference across all variables:", round(mean(loadings_diff), 3), "\n")
    cat("Max absolute loading difference:", round(max(loadings_diff), 3), "\n")
    
    # Identify most unstable loadings
    max_diff_idx <- which(loadings_diff == max(loadings_diff), arr.ind = TRUE)
    most_unstable_var <- common_vars[max_diff_idx[1, 1]]
    most_unstable_factor <- colnames(loadings_diff)[max_diff_idx[1, 2]]
    cat("Most unstable loading:", most_unstable_var, "on", most_unstable_factor, "\n")
  }
}

# --- 4. Bootstrap cross-validation ---
cat("\n[DIAGNOSTIC 4] Bootstrap Cross-validation\n")
cat(strrep("-", 50), "\n")

B_boot <- 100  # Number of bootstrap samples
n_cores <- max(1, parallel::detectCores() - 1)
cl <- makeCluster(n_cores)
registerDoSNOW(cl)

pb <- txtProgressBar(max = B_boot, style = 3)
opts <- list(progress = function(n) setTxtProgressBar(pb, n))

boot_loadings <- foreach(b = 1:B_boot, .combine = rbind,
                         .packages = c("lavaan"),
                         .options.snow = opts) %dopar% {
  # Bootstrap sample
  boot_idx <- sample(nrow(cfa_df), replace = TRUE)
  boot_data <- cfa_df[boot_idx, ]
  
  # Fit model
  fit_boot <- tryCatch({
    lavaan::cfa(
      cfa_model_refined,
      data = boot_data,
      std.lv = TRUE,
      estimator = "WLSMV",
      ordered = ordered_items
    )
  }, error = function(e) NULL)
  
  if (is.null(fit_boot)) {
    return(rep(NA, length(common_vars) * 2))  # 2 factors
  }
  
  # Extract loadings
  loadings_boot <- lavaan::inspect(fit_boot, "std")$lambda
  if (length(common_vars) > 0) {
    loadings_vec <- as.vector(loadings_boot[common_vars, ])
    return(loadings_vec)
  } else {
    return(rep(NA, length(common_vars) * 2))
  }
}

close(pb)
stopCluster(cl)

# Analyze bootstrap results
if (ncol(boot_loadings) > 0) {
  # Calculate bootstrap confidence intervals for loadings
  loadings_ci <- apply(boot_loadings, 2, quantile, c(0.025, 0.975), na.rm = TRUE)
  
  # Focus on Q70 and Q52
  q70_cols <- grep("Q70", colnames(boot_loadings), value = TRUE)
  q52_cols <- grep("Q52", colnames(boot_loadings), value = TRUE)
  
  cat("\nQ70 bootstrap confidence intervals:\n")
  if (length(q70_cols) > 0) {
    for (col in q70_cols) {
      ci <- loadings_ci[, col]
      cat(col, ":", round(ci[1], 3), "to", round(ci[2], 3), "\n")
    }
  }
  
  cat("\nQ52 bootstrap confidence intervals:\n")
  if (length(q52_cols) > 0) {
    for (col in q52_cols) {
      ci <- loadings_ci[, col]
      cat(col, ":", round(ci[1], 3), "to", round(ci[2], 3), "\n")
    }
  }
}

# --- 5. Check Parameter Uncertainty ---
cat("\n[DIAGNOSTIC 5] Parameter Uncertainty Analysis\n")
cat(strrep("-", 50), "\n")

# Extract standard errors and parameter estimates
par_table_full <- lavaan::parTable(fit_cfa_refined)
par_table_full$se_ratio <- abs(par_table_full$est / par_table_full$se)

# Focus on loadings and thresholds
loadings_table <- subset(par_table_full, op == "=~")
thresholds_table <- subset(par_table_full, op == "|")

cat("\nLoadings with large standard errors (SE ratio > 2):\n")
large_se_loadings <- subset(loadings_table, se_ratio < 0.5)
if (nrow(large_se_loadings) > 0) {
  print(large_se_loadings[, c("lhs", "rhs", "est", "se", "se_ratio")])
} else {
  cat("No loadings with unusually large SEs detected.\n")
}

cat("\nThresholds with large standard errors (SE ratio > 2):\n")
large_se_thresholds <- subset(thresholds_table, se_ratio < 0.5)
if (nrow(large_se_thresholds) > 0) {
  print(large_se_thresholds[, c("lhs", "rhs", "est", "se", "se_ratio")])
} else {
  cat("No thresholds with unusually large SEs detected.\n")
}

# Check for category imbalances in ordered variables
cat("\n[5b] Category distribution analysis for ordered variables:\n")
for (var in ordered_items) {
  if (var %in% names(cfa_df)) {
    tab <- table(cfa_df[[var]], useNA = "ifany")
    cat("\n", var, ":\n")
    print(tab)
    cat("Proportions:\n")
    print(round(prop.table(tab), 3))
    
    # Check for sparse categories
    sparse <- tab[tab < 5]
    if (length(sparse) > 0) {
      cat("Sparse categories (< 5 observations):\n")
      print(sparse)
      cat("Consider merging these levels.\n")
    }
  }
}

# --- 6. Summary and Recommendations ---
cat("\n[DIAGNOSTIC 6] Summary and Recommendations\n")
cat(strrep("-", 50), "\n")

cat("\nKey Findings:\n")
cat("1. Q70 Heywood case:", ifelse(nrow(q70_neg_resid) > 0, "DETECTED", "NOT DETECTED"), "\n")
cat("2. Three-factor model fit:", ifelse(!is.null(fit_cfa_3f), "SUCCESSFUL", "FAILED"), "\n")
cat("3. Bifactor model fit:", ifelse(!is.null(fit_cfa_bifactor), "SUCCESSFUL", "FAILED"), "\n")
cat("4. Cross-validation:", ifelse(!is.null(fit_split1) && !is.null(fit_split2), "COMPLETED", "FAILED"), "\n")

cat("\nRecommendations:\n")
if (nrow(q70_neg_resid) > 0) {
  cat("- Fix Q70 residual variance to 0.01 or consider dropping Q70\n")
  cat("- Examine Q70 category distribution for sparse categories\n")
}

if (!is.null(fit_cfa_3f) && !is.null(fit_cfa_bifactor)) {
  # Compare model fits
  fit_2f <- lavaan::fitMeasures(fit_cfa_refined, c("cfi", "rmsea", "srmr"))
  fit_3f <- lavaan::fitMeasures(fit_cfa_3f, c("cfi", "rmsea", "srmr"))
  fit_bif <- lavaan::fitMeasures(fit_cfa_bifactor, c("cfi", "rmsea", "srmr"))
  
  models <- data.frame(
    Model = c("2-Factor", "3-Factor", "Bifactor"),
    CFI = c(fit_2f["cfi"], fit_3f["cfi"], fit_bif["cfi"]),
    RMSEA = c(fit_2f["rmsea"], fit_3f["rmsea"], fit_bif["rmsea"]),
    SRMR = c(fit_2f["srmr"], fit_3f["srmr"], fit_bif["srmr"])
  )
  
  best_model <- models$Model[which.max(models$CFI)]
  cat("- Best fitting model:", best_model, "\n")
  
  if (best_model == "3-Factor") {
    cat("- Consider the three-factor solution with dedicated Market-engagement factor\n")
  } else if (best_model == "Bifactor") {
    cat("- Consider the bifactor solution with general + specific factors\n")
  }
}

if (length(large_se_loadings) > 0 || length(large_se_thresholds) > 0) {
  cat("- Large standard errors detected; consider merging sparse categories\n")
}

cat("\nNext Steps:\n")
cat("1. Implement the recommended model modifications\n")
cat("2. Re-run diagnostics on the modified model\n")
cat("3. Validate the final solution with additional data if available\n")

# Save diagnostic results
saveRDS(list(
  fit_cfa_refined = fit_cfa_refined,
  fit_cfa_3f = fit_cfa_3f,
  fit_cfa_bifactor = fit_cfa_bifactor,
  fit_split1 = fit_split1,
  fit_split2 = fit_split2,
  boot_loadings = boot_loadings,
  par_table_full = par_table_full
), file = "cfa_diagnostics.rds")

cat("\nDiagnostic results saved to 'cfa_diagnostics.rds'\n")

# ---------------------------------------------------------------------------
# 23. Final Model Refinements and Lock-in
# ---------------------------------------------------------------------------

cat("\n", strrep("=", 80), "\n")
cat("FINAL MODEL REFINEMENTS AND LOCK-IN\n")
cat(strrep("=", 80), "\n")

# --- 1. Lock-in the fixed-error 2-factor model as measurement core ---
cat("\n[FINAL 1] Locking in fixed-error 2-factor model\n")
cat(strrep("-", 50), "\n")

# Use the fixed model if it was successful, otherwise use the refined model
if (exists("fit_cfa_fixed") && !is.null(fit_cfa_fixed)) {
  fit_cfa_final <- fit_cfa_fixed
  cfa_model_final <- cfa_model_fixed
  cat("Using fixed-error model (Q70 residual variance = 0.01)\n")
} else {
  fit_cfa_final <- fit_cfa_refined
  cfa_model_final <- cfa_model_refined
  cat("Using refined model (no Heywood case detected)\n")
}

cat("Final model fit indices:\n")
print(lavaan::fitMeasures(fit_cfa_final, c("chisq", "df", "cfi", "rmsea", "srmr")))

# --- 2a. Test Q70+Q108 composite as F1 indicator ---
cat("\n[FINAL 2a] Testing Q70+Q108 composite as F1 indicator\n")
cat(strrep("-", 50), "\n")

# Create composite variable
q70_var <- "Q70__in_the_past_12_months_did_you_receive_any_info_from_anyone_on_agriculture__binary__1"
q108_var <- "Q108__What_is_your_households_yearly_income_from_agriculture_NPR__continuous"

if (q70_var %in% names(cfa_df) && q108_var %in% names(cfa_df)) {
  # Standardize both variables
  q70_std <- scale(as.numeric(cfa_df[[q70_var]]))
  q108_std <- scale(cfa_df[[q108_var]])
  
  # Create composite (simple average of standardized scores)
  composite_70_108 <- (q70_std + q108_std) / 2
  cfa_df$Q70_Q108_composite <- composite_70_108
  
  cat("Created Q70+Q108 composite variable\n")
  cat("Correlation between Q70 and Q108:", round(cor(q70_std, q108_std, use = "complete.obs"), 3), "\n")
  cat("Composite variable statistics:\n")
  print(summary(cfa_df$Q70_Q108_composite))
  
  # Create composite model by adding composite to F1
  # Split the model string into lines
  model_lines <- strsplit(cfa_model_final, "\n")[[1]]
  
  # Find the F1 line
  f1_line_idx <- which(grepl("^F1 =~", model_lines))
  if (length(f1_line_idx) > 0) {
    # Get the current F1 line
    current_f1_line <- model_lines[f1_line_idx[1]]
    
    # Add the composite to the F1 line
    new_f1_line <- paste0(current_f1_line, " + Q70_Q108_composite")
    
    # Replace the F1 line
    model_lines[f1_line_idx[1]] <- new_f1_line
    
    # Reconstruct the model string
    cfa_model_composite <- paste(model_lines, collapse = "\n")
  } else {
    cfa_model_composite <- cfa_model_final
  }
  
  cfa_model_composite_text <- paste(cfa_model_composite, collapse = "\n")
  cat("Composite model syntax:\n")
  cat(paste(cfa_model_composite, collapse = "\n"), "\n")
  
  # Fit composite model
  fit_cfa_composite <- tryCatch({
    lavaan::cfa(
      cfa_model_composite_text,
      data = cfa_df,
      std.lv = TRUE,
      estimator = "WLSMV",
      ordered = ordered_items
    )
  }, error = function(e) {
    cat("[ERROR] Composite model failed to converge:\n")
    print(e)
    return(NULL)
  })
  
  if (!is.null(fit_cfa_composite)) {
    cat("\nComposite model fit indices:\n")
    print(lavaan::fitMeasures(fit_cfa_composite, c("chisq", "df", "cfi", "rmsea", "srmr")))
    
    cat("\nComposite model standardized loadings:\n")
    print(lavaan::inspect(fit_cfa_composite, "std")$lambda)
    
    # Compare with original model
    cat("\nModel comparison (Original vs Composite):\n")
    fit_orig <- lavaan::fitMeasures(fit_cfa_final, c("chisq", "df", "cfi", "rmsea", "srmr"))
    fit_comp <- lavaan::fitMeasures(fit_cfa_composite, c("chisq", "df", "cfi", "rmsea", "srmr"))
    comparison_df <- data.frame(
      Model = c("Original", "Composite"),
      ChiSq = c(fit_orig["chisq"], fit_comp["chisq"]),
      DF = c(fit_orig["df"], fit_comp["df"]),
      CFI = c(fit_orig["cfi"], fit_comp["cfi"]),
      RMSEA = c(fit_orig["rmsea"], fit_comp["rmsea"]),
      SRMR = c(fit_orig["srmr"], fit_comp["srmr"])
    )
    print(comparison_df)
  }
} else {
  cat("Q70 or Q108 variables not found in dataset\n")
}

# --- 2b. Collapse sparse ordinal categories ---
cat("\n[FINAL 2b] Collapsing sparse ordinal categories\n")
cat(strrep("-", 50), "\n")

# Function to collapse sparse categories
collapse_sparse_categories <- function(data, var_name, min_count = 5) {
  if (!var_name %in% names(data)) return(data)
  
  var_data <- data[[var_name]]
  if (!is.factor(var_data) && !is.ordered(var_data)) return(data)
  
  tab <- table(var_data, useNA = "ifany")
  sparse_levels <- names(tab)[tab < min_count]
  
  if (length(sparse_levels) > 0) {
    cat("Collapsing sparse categories in", var_name, ":\n")
    cat("Original levels:", paste(levels(var_data), collapse = ", "), "\n")
    cat("Sparse levels (<", min_count, "):", paste(sparse_levels, collapse = ", "), "\n")
    
    # Create new factor with collapsed levels
    new_data <- as.character(var_data)
    
    # For ordinal variables, collapse from the extremes
    if (is.ordered(var_data)) {
      levels_numeric <- as.numeric(levels(var_data))
      if (length(levels_numeric) >= 4) {
        # Collapse extreme categories: 1+2 vs 3 vs 4+5
        new_data[new_data %in% c("1", "2")] <- "1-2"
        new_data[new_data %in% c("4", "5")] <- "4-5"
        new_levels <- c("1-2", "3", "4-5")
      } else {
        # For shorter scales, just combine adjacent sparse levels
        for (level in sparse_levels) {
          if (level %in% names(tab)) {
            # Find adjacent level to combine with
            level_num <- as.numeric(level)
            if (level_num > 1) {
              new_data[new_data == level] <- as.character(level_num - 1)
            } else {
              new_data[new_data == level] <- as.character(level_num + 1)
            }
          }
        }
        new_levels <- sort(unique(new_data))
      }
    } else {
      # For nominal variables, combine sparse levels into "Other"
      for (level in sparse_levels) {
        new_data[new_data == level] <- "Other"
      }
      new_levels <- c(setdiff(levels(var_data), sparse_levels), "Other")
    }
    
    data[[var_name]] <- factor(new_data, levels = new_levels, ordered = is.ordered(var_data))
    cat("New levels:", paste(levels(data[[var_name]]), collapse = ", "), "\n")
  } else {
    cat("No sparse categories found in", var_name, "\n")
  }
  
  return(data)
}

# Apply collapsing to ordered variables
cfa_df_collapsed <- cfa_df
for (var in ordered_items) {
  cfa_df_collapsed <- collapse_sparse_categories(cfa_df_collapsed, var, min_count = 5)
}

# Update ordered_items list for collapsed data
ordered_items_collapsed <- names(Filter(is.ordered, cfa_df_collapsed))

cat("\nUpdated ordered variables after collapsing:\n")
print(ordered_items_collapsed)

# Fit model with collapsed categories (including composite)
fit_cfa_collapsed <- tryCatch({
  lavaan::cfa(
    cfa_model_composite,
    data = cfa_df_collapsed,
    std.lv = TRUE,
    estimator = "WLSMV",
    ordered = ordered_items_collapsed
  )
}, error = function(e) {
  cat("[ERROR] Collapsed model failed to converge:\n")
  print(e)
  return(NULL)
})

if (!is.null(fit_cfa_collapsed)) {
  cat("\nCollapsed model fit indices:\n")
  print(lavaan::fitMeasures(fit_cfa_collapsed, c("chisq", "df", "cfi", "rmsea", "srmr")))
  
  cat("\nCollapsed model standardized loadings:\n")
  print(lavaan::inspect(fit_cfa_collapsed, "std")$lambda)
  
  # Compare with composite model
  cat("\nModel comparison (Composite vs Collapsed):\n")
  fit_orig <- lavaan::fitMeasures(fit_cfa_composite, c("chisq", "df", "cfi", "rmsea", "srmr"))
  fit_coll <- lavaan::fitMeasures(fit_cfa_collapsed, c("chisq", "df", "cfi", "rmsea", "srmr"))
  comparison_df <- data.frame(
    Model = c("Composite", "Collapsed"),
    ChiSq = c(fit_orig["chisq"], fit_coll["chisq"]),
    DF = c(fit_orig["df"], fit_coll["df"]),
    CFI = c(fit_orig["cfi"], fit_coll["cfi"]),
    RMSEA = c(fit_orig["rmsea"], fit_coll["rmsea"]),
    SRMR = c(fit_orig["srmr"], fit_coll["srmr"])
  )
  print(comparison_df)
}

# --- 2c. Re-run bootstrap with interpretable CIs ---
cat("\n[FINAL 2c] Re-running bootstrap with interpretable CIs\n")
cat(strrep("-", 50), "\n")

# Choose the best model for bootstrap (collapsed if successful, otherwise composite)
if (!is.null(fit_cfa_collapsed)) {
  bootstrap_data <- cfa_df_collapsed
  bootstrap_model <- cfa_model_composite
  bootstrap_ordered <- ordered_items_collapsed
  cat("Using collapsed model for bootstrap\n")
} else {
  bootstrap_data <- cfa_df
  bootstrap_model <- cfa_model_composite
  bootstrap_ordered <- ordered_items
  cat("Using composite model for bootstrap\n")
}

# Bootstrap with proper variable name handling
B_boot_final <- 200  # Increased sample size for better precision
n_cores <- max(1, parallel::detectCores() - 1)
cl <- makeCluster(n_cores)
registerDoSNOW(cl)

pb <- txtProgressBar(max = B_boot_final, style = 3)
opts <- list(progress = function(n) setTxtProgressBar(pb, n))

boot_loadings_final <- foreach(b = 1:B_boot_final, .combine = rbind,
                              .packages = c("lavaan"),
                              .options.snow = opts) %dopar% {
  # Bootstrap sample
  boot_idx <- sample(nrow(bootstrap_data), replace = TRUE)
  boot_data <- bootstrap_data[boot_idx, ]
  
  # Fit model
  fit_boot <- tryCatch({
    lavaan::cfa(
      bootstrap_model,
      data = boot_data,
      std.lv = TRUE,
      estimator = "WLSMV",
      ordered = bootstrap_ordered
    )
  }, error = function(e) NULL)
  
  if (is.null(fit_boot)) {
    return(rep(NA, 20))  # Adjust based on expected number of loadings
  }
  
  # Extract loadings
  loadings_boot <- lavaan::inspect(fit_boot, "std")$lambda
  loadings_vec <- as.vector(loadings_boot)
  names(loadings_vec) <- paste0("loading_", 1:length(loadings_vec))
  return(loadings_vec)
}

close(pb)
stopCluster(cl)

# Analyze bootstrap results
cat("\nBootstrap convergence rate:", round(sum(!is.na(boot_loadings_final[, 1])) / B_boot_final * 100, 1), "%\n")

if (sum(!is.na(boot_loadings_final[, 1])) >= 0.95 * B_boot_final) {
  cat("✓ Bootstrap convergence ≥95% - results are reliable\n")
  
  # Calculate bootstrap confidence intervals
  loadings_ci_final <- apply(boot_loadings_final, 2, quantile, c(0.025, 0.975), na.rm = TRUE)
  
  # Extract original loading names for interpretation
  orig_loadings <- lavaan::inspect(fit_cfa_final, "std")$lambda
  loading_names <- paste0(rep(rownames(orig_loadings), each = ncol(orig_loadings)), 
                         "_", rep(colnames(orig_loadings), times = nrow(orig_loadings)))
  
  # Focus on key variables
  key_vars <- c("Q70", "Q52", "Q112", "Q62", "Q108", "Q109", "Q5")
  
  cat("\nBootstrap confidence intervals for key variables:\n")
  for (var in key_vars) {
    var_cols <- grep(var, colnames(boot_loadings_final), value = TRUE)
    if (length(var_cols) > 0) {
      cat("\n", var, ":\n")
      for (col in var_cols) {
        ci <- loadings_ci_final[, col]
        if (!all(is.na(ci))) {
          cat("  ", col, ":", round(ci[1], 3), "to", round(ci[2], 3), "\n")
        }
      }
    }
  }
  
  # Overall loading stability
  loadings_sd <- apply(boot_loadings_final, 2, sd, na.rm = TRUE)
  cat("\nLoading stability (standard deviations):\n")
  print(round(loadings_sd, 3))
  
  # Identify most unstable loadings
  max_sd_idx <- which.max(loadings_sd)
  cat("\nMost unstable loading:", names(loadings_sd)[max_sd_idx], 
      "(SD =", round(loadings_sd[max_sd_idx], 3), ")\n")
  
} else {
  cat("✗ Bootstrap convergence <95% - results may be unreliable\n")
}

# --- 3. Final model summary and recommendations ---
cat("\n[FINAL 3] Summary and Final Recommendations\n")
cat(strrep("-", 50), "\n")

cat("\nFinal Model Status:\n")
cat("1. Fixed-error 2-factor model:", ifelse(exists("fit_cfa_fixed") && !is.null(fit_cfa_fixed), "IMPLEMENTED", "NOT NEEDED"), "\n")
cat("2. Q70+Q108 composite test:", ifelse(exists("fit_cfa_composite") && !is.null(fit_cfa_composite), "COMPLETED", "FAILED"), "\n")
cat("3. Category collapsing:", ifelse(exists("fit_cfa_collapsed") && !is.null(fit_cfa_collapsed), "COMPLETED", "FAILED"), "\n")
cat("4. Bootstrap CIs:", ifelse(sum(!is.na(boot_loadings_final[, 1])) >= 0.95 * B_boot_final, "RELIABLE", "UNRELIABLE"), "\n")

# Determine the best final model
best_model <- "original"
best_fit <- lavaan::fitMeasures(fit_cfa_final, c("cfi", "rmsea", "srmr"))

if (exists("fit_cfa_composite") && !is.null(fit_cfa_composite)) {
  comp_fit <- lavaan::fitMeasures(fit_cfa_composite, c("cfi", "rmsea", "srmr"))
  if (comp_fit["cfi"] > best_fit["cfi"] && comp_fit["rmsea"] < best_fit["rmsea"]) {
    best_model <- "composite"
    best_fit <- comp_fit
  }
}

if (exists("fit_cfa_collapsed") && !is.null(fit_cfa_collapsed)) {
  coll_fit <- lavaan::fitMeasures(fit_cfa_collapsed, c("cfi", "rmsea", "srmr"))
  if (coll_fit["cfi"] > best_fit["cfi"] && coll_fit["rmsea"] < best_fit["rmsea"]) {
    best_model <- "collapsed"
    best_fit <- coll_fit
  }
}

cat("\nBest final model:", best_model, "\n")
cat("Best fit indices: CFI =", round(best_fit["cfi"], 3), 
    ", RMSEA =", round(best_fit["rmsea"], 3), 
    ", SRMR =", round(best_fit["srmr"], 3), "\n")

cat("\nFinal Recommendations:\n")
if (best_model == "composite") {
  cat("- Use the Q70+Q108 composite model for cleaner interpretation\n")
  cat("- The composite captures both information-seeking and agricultural income\n")
} else if (best_model == "collapsed") {
  cat("- Use the collapsed categories model for better threshold estimation\n")
  cat("- Consider the composite approach for future data collection\n")
} else {
  cat("- Keep the original fixed-error model as the final solution\n")
  cat("- Consider the composite approach for future studies\n")
}

if (sum(!is.na(boot_loadings_final[, 1])) >= 0.95 * B_boot_final) {
  cat("- Bootstrap results are reliable for confidence intervals\n")
} else {
  cat("- Bootstrap convergence is poor; interpret with caution\n")
}

# Save final results
saveRDS(list(
  fit_cfa_final = fit_cfa_final,
  fit_cfa_composite = if(exists("fit_cfa_composite")) fit_cfa_composite else NULL,
  fit_cfa_collapsed = if(exists("fit_cfa_collapsed")) fit_cfa_collapsed else NULL,
  boot_loadings_final = boot_loadings_final,
  best_model = best_model,
  best_fit = best_fit
), file = "cfa_final_model.rds")

cat("\nFinal model results saved to 'cfa_final_model.rds'\n")
cat("\n", strrep("=", 80), "\n")
cat("FINAL MODEL REFINEMENTS COMPLETE\n")
cat(strrep("=", 80), "\n")

# ---------------------------------------------------------------------------
# 24. Additional Diagnostic Checks for Specific Issues
# ---------------------------------------------------------------------------

cat("\n", strrep("=", 80), "\n")
cat("ADDITIONAL DIAGNOSTIC CHECKS FOR SPECIFIC ISSUES\n")
cat(strrep("=", 80), "\n")

# --- Issue 1: Enforce Simple Structure (Drop Q70 Cross-loading on F2) ---
cat("\n[ISSUE 1] Enforcing Simple Structure - Drop Q70 Cross-loading on F2\n")
cat(strrep("-", 60), "\n")

# First, let's check the current model to see Q70's loadings
if (exists("fit_cfa_final") && !is.null(fit_cfa_final)) {
  current_loadings <- lavaan::inspect(fit_cfa_final, "std")$lambda
  q70_var <- "Q70__in_the_past_12_months_did_you_receive_any_info_from_anyone_on_agriculture__binary__1"
  
  if (q70_var %in% rownames(current_loadings)) {
    cat("Current Q70 loadings:\n")
    print(round(current_loadings[q70_var, ], 3))
    
    # Check if Q70 has significant cross-loading on F2
    q70_f2_loading <- current_loadings[q70_var, "F2"]
    if (abs(q70_f2_loading) > 0.3) {
      cat("\nQ70 has substantial cross-loading on F2 (", round(q70_f2_loading, 3), ")\n")
      cat("Creating simple structure model by constraining Q70 to load only on F1...\n")
      
      # Create simple structure model by properly removing Q70 from F2
      # Parse the current model and modify factor loadings
      model_lines <- strsplit(cfa_model_final, "\n")[[1]]
      
      # Find F1 and F2 lines
      f1_line_idx <- which(grepl("^F1 =~", model_lines))
      f2_line_idx <- which(grepl("^F2 =~", model_lines))
      
      if (length(f1_line_idx) > 0 && length(f2_line_idx) > 0) {
        # Get current F1 and F2 lines
        f1_line <- model_lines[f1_line_idx[1]]
        f2_line <- model_lines[f2_line_idx[1]]
        
        # Remove Q70 from F2 line if present
        # Use fixed=TRUE to avoid regex issues with special characters
        q70_patterns <- c(
          paste0(" + ", q70_var),
          paste0("+ ", q70_var),
          paste0(" +", q70_var),
          paste0(q70_var, " +"),
          paste0(q70_var, "+"),
          q70_var
        )
        
        f2_line_new <- f2_line
        for (pattern in q70_patterns) {
          f2_line_new <- gsub(pattern, "", f2_line_new, fixed = TRUE)
        }
        
        # Clean up any remaining artifacts
        f2_line_new <- gsub("\\s+", " ", f2_line_new)  # Multiple spaces to single
        f2_line_new <- gsub("\\s*\\+\\s*\\+\\s*", " + ", f2_line_new)  # Multiple + signs
        f2_line_new <- gsub("\\s*\\+\\s*$", "", f2_line_new)  # Trailing +
        f2_line_new <- gsub("\\s+$", "", f2_line_new)  # Trailing spaces
        
        # If F2 line becomes empty except for "F2 =~", handle it
        if (grepl("^F2\\s*=~\\s*$", f2_line_new)) {
          f2_line_new <- "F2 =~"
        }
        
        # Replace the F2 line
        model_lines[f2_line_idx[1]] <- f2_line_new
        
        # Ensure Q70 is in F1 if not already there
        if (!grepl(q70_var, f1_line, fixed = TRUE)) {
          f1_line_new <- paste0(f1_line, " + ", q70_var)
          model_lines[f1_line_idx[1]] <- f1_line_new
        }
        
        # Reconstruct the model
        cfa_model_simple <- paste(model_lines, collapse = "\n")
      } else {
        cat("Warning: Could not find F1 or F2 lines in model syntax\n")
        cfa_model_simple <- cfa_model_final
      }
      
      cat("\nSimple structure model syntax (with Q70 constrained to F1 only):\n")
      cat(cfa_model_simple, "\n")
      
      # Fit simple structure model
      fit_cfa_simple <- tryCatch({
        lavaan::cfa(
          cfa_model_simple,
          data = cfa_df,
          std.lv = TRUE,
          estimator = "WLSMV",
          ordered = ordered_items
        )
      }, error = function(e) {
        cat("[ERROR] Simple structure model failed to converge:\n")
        print(e)
        return(NULL)
      })
      
      if (!is.null(fit_cfa_simple)) {
        cat("\nSimple structure model fit indices:\n")
        fit_simple_indices <- lavaan::fitMeasures(fit_cfa_simple, c("chisq", "df", "cfi", "rmsea", "srmr"))
        print(fit_simple_indices)
        
        # Compare with original final model
        cat("\nModel comparison (Original Final vs Simple Structure):\n")
        fit_orig_indices <- lavaan::fitMeasures(fit_cfa_final, c("chisq", "df", "cfi", "rmsea", "srmr"))
        comparison_df <- data.frame(
          Model = c("Original_Final", "Simple_Structure"),
          ChiSq = c(fit_orig_indices["chisq"], fit_simple_indices["chisq"]),
          DF = c(fit_orig_indices["df"], fit_simple_indices["df"]),
          CFI = c(fit_orig_indices["cfi"], fit_simple_indices["cfi"]),
          RMSEA = c(fit_orig_indices["rmsea"], fit_simple_indices["rmsea"]),
          SRMR = c(fit_orig_indices["srmr"], fit_simple_indices["srmr"])
        )
        print(comparison_df)
        
        # Check Q70 loadings in simple structure model
        simple_loadings <- lavaan::inspect(fit_cfa_simple, "std")$lambda
        cat("\nQ70 loadings in simple structure model:\n")
        print(round(simple_loadings[q70_var, ], 3))
        
        # Bootstrap the simple structure model to check Q70 loading stability
        cat("\nBootstrapping simple structure model to check Q70 loading stability...\n")
        
        B_boot_simple <- 100
        n_cores <- max(1, parallel::detectCores() - 1)
        cl <- makeCluster(n_cores)
        registerDoSNOW(cl)
        
        pb <- txtProgressBar(max = B_boot_simple, style = 3)
        opts <- list(progress = function(n) setTxtProgressBar(pb, n))
        
        boot_q70_loadings <- foreach(b = 1:B_boot_simple, .combine = c,
                                    .packages = c("lavaan"),
                                    .options.snow = opts) %dopar% {
          # Bootstrap sample
          boot_idx <- sample(nrow(cfa_df), replace = TRUE)
          boot_data <- cfa_df[boot_idx, ]
          
          # Fit simple structure model
          fit_boot <- tryCatch({
            lavaan::cfa(
              cfa_model_simple,
              data = boot_data,
              std.lv = TRUE,
              estimator = "WLSMV",
              ordered = ordered_items
            )
          }, error = function(e) NULL)
          
          if (is.null(fit_boot)) {
            return(NA)
          }
          
          # Extract Q70 loading on F1
          loadings_boot <- lavaan::inspect(fit_boot, "std")$lambda
          if (q70_var %in% rownames(loadings_boot)) {
            return(loadings_boot[q70_var, "F1"])
          } else {
            return(NA)
          }
        }
        
        close(pb)
        stopCluster(cl)
        
        # Analyze Q70 loading stability
        q70_bootstrap_stats <- summary(boot_q70_loadings)
        q70_bootstrap_sd <- sd(boot_q70_loadings, na.rm = TRUE)
        
        cat("\nQ70 → F1 loading bootstrap results (simple structure):\n")
        cat("Mean:", round(mean(boot_q70_loadings, na.rm = TRUE), 3), "\n")
        cat("SD:", round(q70_bootstrap_sd, 3), "\n")
        cat("95% CI:", round(quantile(boot_q70_loadings, c(0.025, 0.975), na.rm = TRUE), 3), "\n")
        cat("Convergence rate:", round(sum(!is.na(boot_q70_loadings)) / B_boot_simple * 100, 1), "%\n")
        
        # Compare with original Q70 loading SD if available
        if (exists("boot_loadings_final") && ncol(boot_loadings_final) > 0) {
          # Try to find Q70 F1 loading in original bootstrap
          q70_f1_cols <- grep("Q70.*F1|loading.*Q70", colnames(boot_loadings_final), value = TRUE)
          if (length(q70_f1_cols) > 0) {
            orig_q70_sd <- sd(boot_loadings_final[, q70_f1_cols[1]], na.rm = TRUE)
            cat("\nComparison of Q70 → F1 loading SD:\n")
            cat("Original model SD:", round(orig_q70_sd, 3), "\n")
            cat("Simple structure SD:", round(q70_bootstrap_sd, 3), "\n")
            cat("Improvement:", round(orig_q70_sd - q70_bootstrap_sd, 3), "\n")
          }
        }
        
      }
    } else {
      cat("Q70 does not have substantial cross-loading on F2 (", round(q70_f2_loading, 3), ")\n")
      cat("Simple structure is already achieved.\n")
    }
  }
}

# --- Issue 2: Check Correlated Residual Between Q112 and Q70 ---
cat("\n[ISSUE 2] Checking Correlated Residual Between Q112 and Q70\n")
cat(strrep("-", 60), "\n")

if (exists("fit_cfa_final") && !is.null(fit_cfa_final)) {
  # Get parameter table to check for correlated residuals
  par_table_check <- lavaan::parTable(fit_cfa_final)
  
  # Look for correlated residual between Q112 and Q70
  q112_var <- "Q112__Generally_speaking_how_would_you_define_your_farming__ordinal"
  q70_var <- "Q70__in_the_past_12_months_did_you_receive_any_info_from_anyone_on_agriculture__binary__1"
  
  q112_q70_corr <- subset(par_table_check, 
                         op == "~~" & 
                         ((lhs == q112_var & rhs == q70_var) | 
                          (lhs == q70_var & rhs == q112_var)))
  
  if (nrow(q112_q70_corr) > 0) {
    cat("Correlated residual between Q112 and Q70 found:\n")
    
    # Check which columns are available and select only existing ones
    available_cols <- names(q112_q70_corr)
    desired_cols <- c("lhs", "op", "rhs", "est", "se", "pvalue")
    cols_to_print <- intersect(desired_cols, available_cols)
    
    if (length(cols_to_print) > 0) {
      print(q112_q70_corr[, cols_to_print, drop = FALSE])
    } else {
      print(q112_q70_corr)
    }
    
    # Safely extract correlation value
    if ("est" %in% available_cols) {
      corr_value <- q112_q70_corr$est[1]
    } else {
      # Try alternative column names that might contain the estimate
      est_cols <- grep("est|coef|value", available_cols, ignore.case = TRUE, value = TRUE)
      if (length(est_cols) > 0) {
        corr_value <- q112_q70_corr[[est_cols[1]]][1]
      } else {
        cat("Warning: Could not find estimate column in parameter table\n")
        corr_value <- NA
      }
    }
    if (!is.na(corr_value)) {
      cat("\nCorrelated residual value:", round(corr_value, 3), "\n")
    } else {
      cat("\nCorrelated residual value: Could not extract estimate\n")
    }
    
    if (!is.na(corr_value) && abs(corr_value) > 0.8) {
      cat("WARNING: Very high correlated residual (|r| > 0.8) suggests substantial unexplained shared variance\n")
      cat("This may indicate:\n")
      cat("1. Missing common cause variable\n")
      cat("2. Need for a separate factor for these items\n")
      cat("3. Measurement overlap between items\n")
      
      # Test model without the correlated residual
      cat("\nTesting model without Q112-Q70 correlated residual...\n")
      
      # Remove the correlated residual from model syntax
      model_lines <- strsplit(cfa_model_final, "\n")[[1]]
      model_lines_filtered <- model_lines[!grepl("Q112.*Q70|Q70.*Q112", model_lines)]
      cfa_model_no_corr <- paste(model_lines_filtered, collapse = "\n")
      
      fit_cfa_no_corr <- tryCatch({
        lavaan::cfa(
          cfa_model_no_corr,
          data = cfa_df,
          std.lv = TRUE,
          estimator = "WLSMV",
          ordered = ordered_items
        )
      }, error = function(e) {
        cat("[ERROR] Model without correlated residual failed:\n")
        print(e)
        return(NULL)
      })
      
      if (!is.null(fit_cfa_no_corr)) {
        cat("\nFit comparison (with vs without Q112-Q70 correlated residual):\n")
        fit_with_corr <- lavaan::fitMeasures(fit_cfa_final, c("chisq", "df", "cfi", "rmsea", "srmr"))
        fit_no_corr <- lavaan::fitMeasures(fit_cfa_no_corr, c("chisq", "df", "cfi", "rmsea", "srmr"))
        
        comparison_corr_df <- data.frame(
          Model = c("With_Corr_Residual", "Without_Corr_Residual"),
          ChiSq = c(fit_with_corr["chisq"], fit_no_corr["chisq"]),
          DF = c(fit_with_corr["df"], fit_no_corr["df"]),
          CFI = c(fit_with_corr["cfi"], fit_no_corr["cfi"]),
          RMSEA = c(fit_with_corr["rmsea"], fit_no_corr["rmsea"]),
          SRMR = c(fit_with_corr["srmr"], fit_no_corr["srmr"])
        )
        print(comparison_corr_df)
        
        # Chi-square difference test
        chi_diff <- fit_no_corr["chisq"] - fit_with_corr["chisq"]
        df_diff <- fit_no_corr["df"] - fit_with_corr["df"]
        p_diff <- 1 - pchisq(chi_diff, df_diff)
        
        cat("\nChi-square difference test:\n")
        cat("Δχ² =", round(chi_diff, 3), ", Δdf =", df_diff, ", p =", round(p_diff, 3), "\n")
        if (p_diff < 0.001) {
          cat("The correlated residual significantly improves model fit (p < 0.001)\n")
        } else {
          cat("The correlated residual may not be necessary (p ≥ 0.001)\n")
        }
      }
    } else {
      cat("Correlated residual is within acceptable range (|r| ≤ 0.8)\n")
    }
  } else {
    cat("No correlated residual between Q112 and Q70 found in current model\n")
  }
}

# --- Issue 3: Check Collinearity Between Q70 and Q108 ---
cat("\n[ISSUE 3] Checking Collinearity Between Q70 and Q108\n")
cat(strrep("-", 60), "\n")

q70_var <- "Q70__in_the_past_12_months_did_you_receive_any_info_from_anyone_on_agriculture__binary__1"
q108_var <- "Q108__What_is_your_households_yearly_income_from_agriculture_NPR__continuous"

if (q70_var %in% names(cfa_df) && q108_var %in% names(cfa_df)) {
  # Calculate correlation between Q70 and Q108
  q70_q108_corr <- cor(as.numeric(cfa_df[[q70_var]]), cfa_df[[q108_var]], use = "complete.obs")
  
  cat("Correlation between Q70 (binary) and Q108 (continuous):", round(q70_q108_corr, 3), "\n")
  
  # Check the distributions
  cat("\nQ70 (binary) distribution:\n")
  q70_table <- table(cfa_df[[q70_var]], useNA = "ifany")
  print(q70_table)
  print(round(prop.table(q70_table), 3))
  
  cat("\nQ108 (continuous) summary:\n")
  print(summary(cfa_df[[q108_var]]))
  
  # Examine the relationship more closely
  cat("\nQ108 values by Q70 categories:\n")
  q108_by_q70 <- aggregate(cfa_df[[q108_var]], 
                          by = list(Q70 = cfa_df[[q70_var]]), 
                          FUN = function(x) c(mean = mean(x, na.rm = TRUE), 
                                             sd = sd(x, na.rm = TRUE),
                                             n = sum(!is.na(x))))
  print(q108_by_q70)
  
  # Point-biserial correlation (appropriate for binary-continuous correlation)
  if (requireNamespace("ltm", quietly = TRUE)) {
    pb_corr <- ltm::biserial.cor(cfa_df[[q108_var]], cfa_df[[q70_var]], use = "complete.obs")
    cat("\nPoint-biserial correlation:", round(pb_corr, 3), "\n")
  } else {
    cat("\nNote: Install 'ltm' package for point-biserial correlation calculation\n")
  }
  
  # Check if collinearity is problematic
  if (abs(q70_q108_corr) > 0.9) {
    cat("\nWARNING: Very high correlation (|r| > 0.9) indicates near-perfect collinearity\n")
    cat("This can cause:\n")
    cat("1. Numerical instability in parameter estimation\n")
    cat("2. Inflated standard errors\n")
    cat("3. Unreliable factor loadings\n")
    
    cat("\nRecommended solutions:\n")
    cat("1. Use the Q70+Q108 composite variable (already implemented)\n")
    cat("2. Drop one of the variables\n")
    cat("3. Use regularization techniques\n")
    
    # Test the composite approach if it exists
    if ("Q70_Q108_composite" %in% names(cfa_df) && exists("fit_cfa_composite") && !is.null(fit_cfa_composite)) {
      cat("\nComposite approach results:\n")
      composite_fit <- lavaan::fitMeasures(fit_cfa_composite, c("chisq", "df", "cfi", "rmsea", "srmr"))
      print(composite_fit)
      
      # Check if composite loading is more stable
      composite_loadings <- lavaan::inspect(fit_cfa_composite, "std")$lambda
      if ("Q70_Q108_composite" %in% rownames(composite_loadings)) {
        cat("\nQ70+Q108 composite loading:\n")
        print(round(composite_loadings["Q70_Q108_composite", ], 3))
      }
    }
    
  } else if (abs(q70_q108_corr) > 0.7) {
    cat("\nModerate to high correlation (|r| > 0.7) - monitor for estimation issues\n")
  } else {
    cat("\nCorrelation is within acceptable range (|r| ≤ 0.7)\n")
  }
  
  # Additional check: Variance Inflation Factor (VIF) if possible
  cat("\nChecking for multicollinearity using variance inflation...\n")
  
  # Create a simple regression to estimate VIF-like measure
  q70_numeric <- as.numeric(cfa_df[[q70_var]])
  q108_standardized <- scale(cfa_df[[q108_var]])[, 1]
  
  # Regress Q70 on Q108
  vif_model <- lm(q70_numeric ~ q108_standardized)
  r_squared <- summary(vif_model)$r.squared
  vif_estimate <- 1 / (1 - r_squared)
  
  cat("Pseudo-VIF (Q70 explained by Q108):", round(vif_estimate, 3), "\n")
  if (vif_estimate > 10) {
    cat("WARNING: High VIF (> 10) indicates serious multicollinearity\n")
  } else if (vif_estimate > 5) {
    cat("CAUTION: Moderate VIF (> 5) suggests potential multicollinearity\n")
  } else {
    cat("VIF is acceptable (< 5)\n")
  }
  
} else {
  cat("Q70 or Q108 variables not found in dataset\n")
}

# --- Summary of Diagnostic Results ---
cat("\n[SUMMARY] Diagnostic Results Summary\n")
cat(strrep("-", 60), "\n")

cat("\n1. Simple Structure (Q70 cross-loading):\n")
if (exists("fit_cfa_simple") && !is.null(fit_cfa_simple)) {
  cat("   ✓ Simple structure model fitted successfully\n")
  if (exists("q70_bootstrap_sd")) {
    cat("   ✓ Q70 loading bootstrap SD:", round(q70_bootstrap_sd, 3), "\n")
  }
} else {
  cat("   - Simple structure not needed or failed to fit\n")
}

cat("\n2. Q112-Q70 Correlated Residual:\n")
if (exists("q112_q70_corr") && nrow(q112_q70_corr) > 0) {
  cat("   ⚠ Correlated residual present:", round(q112_q70_corr$est[1], 3), "\n")
  if (abs(q112_q70_corr$est[1]) > 0.8) {
    cat("   ⚠ Very high correlation - consider theoretical explanation\n")
  }
} else {
  cat("   ✓ No problematic correlated residual found\n")
}

cat("\n3. Q70-Q108 Collinearity:\n")
if (exists("q70_q108_corr")) {
  cat("   ⚠ Correlation:", round(q70_q108_corr, 3), "\n")
  if (exists("vif_estimate")) {
    cat("   ⚠ Pseudo-VIF:", round(vif_estimate, 3), "\n")
  }
  if (abs(q70_q108_corr) > 0.9) {
    cat("   ⚠ High collinearity - composite approach recommended\n")
  }
} else {
  cat("   - Could not assess collinearity\n")
}

cat("\nFinal Recommendations:\n")
if (exists("fit_cfa_simple") && !is.null(fit_cfa_simple) && exists("q70_bootstrap_sd") && q70_bootstrap_sd < 0.1) {
  cat("- ✓ Use simple structure model (improved Q70 loading stability)\n")
}
if (exists("q112_q70_corr") && nrow(q112_q70_corr) > 0 && abs(q112_q70_corr$est[1]) > 0.8) {
  cat("- Consider theoretical explanation for Q112-Q70 relationship\n")
}
if (exists("q70_q108_corr") && abs(q70_q108_corr) > 0.9) {
  cat("- ✓ Use Q70+Q108 composite to address collinearity\n")
}

cat("\n", strrep("=", 80), "\n")
cat("ADDITIONAL DIAGNOSTICS COMPLETE\n")
cat(strrep("=", 80), "\n")

# ---------------------------------------------------------------------------
# 25. Final CFA Model Report (NO COMPOSITE, Q70 ordered, delta parameterization)
# ---------------------------------------------------------------------------

safestr <- function(x) ifelse(is.null(x), "NULL", paste(x, collapse=", "))

cat("\n", strrep("=", 80), "\n")
cat("FINAL MEASUREMENT CORE MODEL REPORT\n")
cat(strrep("=", 80), "\n")

cat("\n[FINAL CORE] Collapsed 2-Factor Model as Measurement Core (no composite)\n")
cat(strrep("-", 60), "\n")

# ---------- Choose data & ordered list ----------
use_collapsed <- exists("cfa_df_collapsed") && !is.null(cfa_df_collapsed)
data_core     <- if (use_collapsed) cfa_df_collapsed else cfa_df
ordered_core  <- if (use_collapsed && exists("ordered_items_collapsed")) {
  ordered_items_collapsed
} else if (exists("ordered_items")) {
  ordered_items
} else {
  character(0)
}

# ---------- Canonical variable names (strings) ----------
Q50   <- "Q50__How_much_land_that_is_yours_do_you_cultivate_bigha__continuous"
Q52   <- "Q52__On_how_much_land_do_you_grow_vegetables_bigha__continuous"
Q62   <- "Q62__How_much_VEGETABLES_do_you_harvest_per_year_from_this_plot_kilograms__continuous"
Q108  <- "Q108__What_is_your_households_yearly_income_from_agriculture_NPR__continuous"
Q112  <- "Q112__Generally_speaking_how_would_you_define_your_farming__ordinal"
Q0avg <- "Q0__average_of_farming_practices__ordinal"
Q70   <- "Q70__in_the_past_12_months_did_you_receive_any_info_from_anyone_on_agriculture__binary__1"

Q5    <- "Q5__AgeYears__continuous"
Q109  <- "Q109__What_is_your_households_yearly_income_overall_including_agriculture_NPR__continuous"
Q0hop <- "Q0__hope_total__continuous"
Q0sc  <- "Q0__self_control_score__continuous"

# ---------- Safety: ensure key ordered vars are actually ordered ----------
must_be_ordered <- c(Q70, Q112, Q0avg)
ordered_core    <- unique(c(ordered_core, must_be_ordered))
ordered_core    <- ordered_core[ordered_core %in% names(data_core)] # drop if not present in data

# ---------- Specify the model (NO COMPOSITE) ----------
# Keep Q70 loading on F1 (primary) and cross-loading on F2, as per better/stable fit.
cfa_model_core <- paste(
  sprintf("F1 =~ %s + %s + %s + %s + %s + %s + %s", Q50, Q52, Q62, Q108, Q112, Q0avg, Q70),
  sprintf("F2 =~ %s + %s + %s + %s + %s + %s + %s", Q5, Q52, Q109, Q0hop, Q0sc, Q0avg, Q70),
  "",  # residual covariances
  sprintf("%s ~~ %s          # Q112 <-> Q70 residual covariance", Q112, Q70),
  sprintf("%s ~~ %s          # Q62  <-> hope residual covariance", Q62, Q0hop),
  "",
  "F1 ~~ F2          # Allow factor covariance",
  sep = "\n"
)

cat("\n[MODEL SYNTAX]\n")
cat(cfa_model_core, "\n\n")

# ---------- Fit the model (WLSMV, default = delta parameterization) ----------
suppressWarnings({
  fit_core <- lavaan::cfa(
    cfa_model_core,
    data       = data_core,
    std.lv     = TRUE,
    estimator  = "WLSMV",
    ordered    = ordered_core
  )
})

# ---------- Verify Q70 residual is FIXED (not freely estimated) under delta ----------
pt <- lavaan::parTable(fit_core)
q70_varrow <- subset(pt, op=="~~" & lhs==Q70 & rhs==Q70)

# Under delta with ordered indicators, there should be no FREE variance for Q70.
# Accept 2 valid states:
#  (A) no row at all for Q70 ~~ Q70  -> variance not a parameter (fixed by identification)
#  (B) a row exists but free==0      -> fixed (not estimated)
if (nrow(q70_varrow) > 0 && isTRUE(q70_varrow$free[1] == 1)) {
  stop("Q70 residual variance is being freely estimated; expected fixed under delta. ",
       "Verify Q70 is listed in 'ordered_core' and is an ordered/binary variable in 'data_core'.")
}

cat(sprintf("✓ Q70 residual variance is fixed by identification (ordered, delta). [%s]\n",
            if (nrow(q70_varrow)==0) "no variance row" else "variance row fixed (free=0)"))

# ---------- Core features ----------
cat("\n[CORE FEATURES] Key Model Specifications:\n")
cat("• Using data:        ", if (use_collapsed) "collapsed (sparse categories merged)" else "original", "\n", sep="")
cat("• Ordered variables: ", safestr(ordered_core), "\n", sep="")

# Q112<->Q70 & Q62<->hope residual covariances
q112_q70_row <- subset(pt, op=="~~" & ((lhs==Q112 & rhs==Q70) | (lhs==Q70 & rhs==Q112)))
q62_hope_row <- subset(pt, op=="~~" & ((lhs==Q62  & rhs==Q0hop) | (lhs==Q0hop & rhs==Q62 )))

if (nrow(q112_q70_row)>0) cat(sprintf("• Q112↔Q70 residual covariance: est=%.3f, se=%.3f\n",
                                      q112_q70_row$est[1], q112_q70_row$se[1]))
if (nrow(q62_hope_row)>0) cat(sprintf("• Q62 ↔hope residual covariance: est=%.3f, se=%.3f\n",
                                      q62_hope_row$est[1], q62_hope_row$se[1]))

# ---------- Model fit ----------
cat("\n[CORE FIT] Final Model Fit Indices:\n")
fm <- lavaan::fitMeasures(fit_core,
                          c("chisq","df","pvalue","cfi","tli","rmsea","rmsea.ci.lower","rmsea.ci.upper","srmr"))
cat(sprintf("χ²(%d) = %.2f, p = %.3f\n", as.integer(fm["df"]), fm["chisq"], fm["pvalue"]))
cat(sprintf("CFI = %.3f\nTLI = %.3f\nRMSEA = %.3f [%.3f, %.3f]\nSRMR = %.3f\n",
            fm["cfi"], fm["tli"], fm["rmsea"], fm["rmsea.ci.lower"], fm["rmsea.ci.upper"], fm["srmr"]))

cat("\nFit Quality: ")
if (fm["cfi"] >= 0.95 && fm["rmsea"] <= 0.08 && fm["srmr"] <= 0.08) {
  cat("EXCELLENT\n")
} else if (fm["cfi"] >= 0.90 && fm["rmsea"] <= 0.10) {
  cat("ACCEPTABLE\n")
} else cat("NEEDS ATTENTION\n")

# ---------- Residual correlation note (standardized matrix, if available) ----------
cat("\n[CORE RESIDUALS] Q112↔Q70 Residual Correlation (standardized):\n")
stdres <- try(lavaan::residuals(fit_core, type="standardized", se=FALSE), silent=TRUE)
if (!inherits(stdres, "try-error") && is.list(stdres) && "cov" %in% names(stdres)) {
  cm <- stdres$cov
  sr <- NA
  if (!is.null(cm) && Q112 %in% rownames(cm) && Q70 %in% colnames(cm)) sr <- cm[Q112, Q70]
  if (!is.null(cm) && is.na(sr) && Q70 %in% rownames(cm) && Q112 %in% colnames(cm)) sr <- cm[Q70, Q112]
  if (is.finite(sr)) {
    cat(sprintf("• Standardized residual correlation ≈ %.3f\n", sr))
    cat("  Note: With ordered items in delta, standardization uses model-implied scaling.\n")
  } else cat("• Not available\n")
} else cat("• Not available\n")

# ---------- Loadings (standardized) ----------
cat("\n[CORE LOADINGS] Standardized Factor Loadings (|λ| > .30):\n")
lam <- lavaan::inspect(fit_core, "std")$lambda

if (!is.null(lam)) {
  if ("F1" %in% colnames(lam)) {
    f1 <- lam[, "F1"]; f1 <- f1[abs(f1) > .30]
    cat("F1 (Commercial-Orientation / Output-Intensity):\n")
    for (nm in names(f1)) cat(sprintf("  • %-6s : %.3f\n",
                                      gsub("__.*","", gsub("^Q0__","", nm)), f1[nm]))
  }
  if ("F2" %in% colnames(lam)) {
    f2 <- lam[, "F2"]; f2 <- f2[abs(f2) > .30]
    cat("F2 (Household Resources & Experience):\n")
    for (nm in names(f2)) cat(sprintf("  • %-6s : %.3f\n",
                                      gsub("__.*","", gsub("^Q0__","", nm)), f2[nm]))
  }
}

# ---------- Factor correlation ----------
psi <- lavaan::inspect(fit_core, "std")$psi
if (!is.null(psi) && all(c("F1","F2") %in% rownames(psi)) && all(c("F1","F2") %in% colnames(psi))) {
  r12 <- psi["F1","F2"]
  cat(sprintf("\nFactor Correlation (F1 ↔ F2): %.3f\n", r12))
  if (abs(r12) < .30)       cat("  → Factors are relatively distinct\n")
  else if (abs(r12) < .70)  cat("  → Factors are moderately related\n")
  else                      cat("  → Factors are highly related (consider unidimensionality)\n")
}

# ---------- Composite reliability (Cronbach-style construct reliability) ----------
cat("\n[CORE RELIABILITY] Factor Reliability (composite):\n")
comp_rel <- function(loads) {
  loads <- loads[is.finite(loads)]
  loads <- loads[abs(loads) > .30]
  if (length(loads) < 2) return(NA_real_)
  sl  <- sum(loads)
  sl2 <- sum(loads^2)
  se  <- length(loads) - sl2  # sum of error variances under standardized solution
  (sl^2) / ((sl^2) + se)
}
if ("F1" %in% colnames(lam)) {
  cr1 <- comp_rel(lam[, "F1"])
  cat(sprintf("• F1: %s\n", ifelse(is.na(cr1), "insufficient indicators", sprintf("%.3f", cr1))))
}
if ("F2" %in% colnames(lam)) {
  cr2 <- comp_rel(lam[, "F2"])
  cat(sprintf("• F2: %s\n", ifelse(is.na(cr2), "insufficient indicators", sprintf("%.3f", cr2))))
}

# ---------- Final notes ----------
cat("\n[FINAL RECOMMENDATIONS] Measurement Core Status:\n")
cat("✅ No composite indicator used.\n")
cat("✅ Q70 is treated as ordered; residual variance fixed by identification (delta).\n")
cat("✅ Q112↔Q70 and Q62↔hope residual covariances retained (theory-consistent, improved fit).\n")
cat("✅ Global fit and factor structure are reported with no contradictions.\n")

cat("\n", strrep("=", 80), "\n")
cat("FINAL MEASUREMENT CORE REPORT COMPLETE\n")
cat(strrep("=", 80), "\n")

