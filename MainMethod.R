
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
  "mediation",  # causal mediation analysis
  "clue"        # for solve LSAP
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

# Helper: align sign and factor order to a reference (uses clue::solve_LSAP)
align_to_ref <- function(L, L_ref, Phi = NULL) {
  L     <- as.matrix(L); L_ref <- as.matrix(L_ref)
  C     <- psych::factor.congruence(L_ref, L)     # k x k
  perm  <- clue::solve_LSAP(abs(C), maximum = TRUE)
  Lp    <- L[, perm, drop = FALSE]
  sgn   <- sign(diag(t(L_ref) %*% Lp)); sgn[sgn == 0] <- 1
  Lp    <- sweep(Lp, 2, sgn, `*`)
  if (is.null(Phi)) return(list(L = Lp, perm = perm, sgn = sgn))
  Phip  <- Phi[perm, perm, drop = FALSE]
  S     <- diag(sgn)
  Phip  <- S %*% Phip %*% S
  list(L = Lp, Phi = Phip, perm = perm, sgn = sgn)
}

fa_ref <- psych::fa(R, nfactors = k, fm = "minres", rotate = "geominQ", n.obs = nrow(df_num))
L_ref  <- as.matrix(fa_ref$loadings)   # p x k

# Ensure k exists here
stopifnot(exists("k"), is.numeric(k), length(k) == 1L, k >= 1)

# Standardize factor names globally to F1..Fk (prevents MR1/MR2 drift)
colnames(L_ref) <- paste0("F", seq_len(k))

# Bootstrap EFA (Spearman-only)
p <- ncol(df_num)
B <- 1000

n_cores <- max(1, parallel::detectCores() - 1)
cl <- parallel::makeCluster(n_cores); doSNOW::registerDoSNOW(cl)
pb <- txtProgressBar(max = B, style = 3)
opts <- list(progress = function(n) setTxtProgressBar(pb, n))

boot_load <- foreach::foreach(
  b = 1:B, .combine = rbind,
  .packages = c("psych","Matrix","clue"),               # polycor removed
  .export   = c("align_to_ref","L_ref","k")
  , .options.snow = opts
) %dopar% {
  repeat {
    # a) bootstrap sample (size n with replacement)
    samp <- df_num[sample(nrow(df_num), replace = TRUE), , drop = FALSE]
    
    # b) Spearman correlation
    samp_num <- as.data.frame(lapply(samp, as.numeric))
    Rb <- tryCatch(
      cor(samp_num, method = "spearman", use = "pairwise.complete.obs"),
      error = function(e) NULL
    )
    if (is.null(Rb) || any(is.na(Rb))) next
    
    # c) Stabilise correlation (ensure PD)
    Rb <- as.matrix(Matrix::nearPD(Rb, corr = TRUE, keepDiag = TRUE)$mat)
    
    # d) EFA: MINRES + geominQ
    fa_b <- tryCatch(
      psych::fa(Rb, nfactors = k, fm = "minres", rotate = "geominQ", n.obs = nrow(samp)),
      error = function(e) NULL
    )
    if (is.null(fa_b)) next
    
    # e) Align loadings to reference orientation
    L_al <- align_to_ref(fa_b$loadings, L_ref)$L
    
    # f) Return vectorized aligned loadings + uniquenesses
    return(c(as.vector(L_al), fa_b$uniquenesses))
  }
}
close(pb); parallel::stopCluster(cl)

# ---------------------------------------------------------------------------
# Step 11. Summarize bootstrap results
# ---------------------------------------------------------------------------

# Split the stacked bootstrap results into loadings and uniquenesses
lambda_boot <- as.matrix(boot_load[, seq_len(p * k), drop = FALSE])
psi_boot    <- as.matrix(boot_load[, (p * k + 1):(p * k + p), drop = FALSE])

# Item (row) names
item_names <- if (exists("keep_final") && length(keep_final) == p) {
  as.character(keep_final)
} else if (!is.null(rownames(R))) {
  rownames(R)
} else if (!is.null(colnames(R))) {
  colnames(R)
} else {
  paste0("V", seq_len(p))
}

# Factor (column) names — use the standardized F1..Fk
factor_names <- paste0("F", seq_len(k))

# Median loadings (reshape to p x k) and assign dimnames
L_median <- matrix(
  apply(lambda_boot, 2, stats::median, na.rm = TRUE),
  nrow = p, ncol = k, byrow = FALSE
)
dimnames(L_median) <- list(item_names, factor_names)

# Median uniquenesses (Ψ)
psi_median <- apply(psi_boot, 2, stats::median, na.rm = TRUE)
names(psi_median) <- item_names

# Keep a simple alias used elsewhere
vars <- item_names

# 95% CIs for each loading, stacked by factor (use the same factor names)
L_ci <- apply(lambda_boot, 2, stats::quantile, c(.025, .975), na.rm = TRUE)

Fnames <- factor_names  # <- ensure match with L_median colnames
df_L_ci <- do.call(
  rbind,
  lapply(seq_len(k), function(j) {
    cols <- ((j - 1L) * p + 1L):(j * p)
    data.frame(
      variable = vars,
      factor   = rep(Fnames[j], p),   # character, not factor
      lower    = as.numeric(L_ci[1, cols]),
      upper    = as.numeric(L_ci[2, cols]),
      stringsAsFactors = FALSE
    )
  })
)

# ---------------------------------------------------------------------------
# Step 12. Stability-based pruning (KEEP / TENTATIVE / DROP)
# ---------------------------------------------------------------------------

# 12.0 Rebuild aligned 3D array: p x k x B_eff
B_eff <- nrow(lambda_boot)
L_b   <- array(NA_real_, dim = c(p, k, B_eff),
               dimnames = list(vars, paste0("F", 1:k), paste0("b", 1:B_eff)))
for (b in 1:B_eff) {
  L_b[, , b] <- matrix(lambda_boot[b, ], nrow = p, ncol = k,
                       dimnames = list(vars, paste0("F", 1:k)))
}

# 12.1 Choose primary factor per item by median |loading|
median_abs <- apply(abs(L_b), c(1, 2), median)   # p x k
j_star     <- apply(median_abs, 1, which.max)    # length p

# 12.2 Bootstrap series on each item's primary factor
get_series   <- function(i) L_b[i, j_star[i], ]
series_list  <- lapply(seq_len(p), get_series); names(series_list) <- vars

# 12.3 Primary-factor stability across bootstraps
primary_idx  <- apply(abs(L_b), c(1, 3), which.max)          # p x B_eff
primary_stab <- sapply(seq_len(p), function(i) mean(primary_idx[i, ] == j_star[i]))

# 12.4 Salience probability & (diagnostic) sign stability
salience_cut  <- 0.30
salience_prob <- sapply(series_list, function(x) mean(abs(x) >= salience_cut, na.rm = TRUE))
sign_stability <- sapply(series_list, function(x) {
  s <- sign(x); s[s == 0] <- 1; max(mean(s == 1, na.rm = TRUE), mean(s == -1, na.rm = TRUE))
})

# 12.5 Median communality proxy (replace with oblique h2 if Φ_b saved)
h2_med <- sapply(seq_len(p), function(i) median(colSums(L_b[i, , ]^2), na.rm = TRUE))

names(salience_prob) <- vars
names(primary_stab)  <- vars
names(h2_med)        <- vars

# 12.6 Thresholds
th_load_med   <- 0.30
th_sal_prob   <- 0.75
th_primary_st <- 0.75
th_h2_keep    <- 0.20
th_h2_drop    <- 0.10
has_MSA <- exists("MSA_i")
th_msa  <- 0.50

# 12.7 Median |loading| on chosen primary (from L_median)
med_abs_primary <- sapply(seq_len(p), function(i) abs(L_median[i, j_star[i]]))

# 12.8 Decisions
keep <- (med_abs_primary >= th_load_med) &
  (salience_prob    >= th_sal_prob) &
  (primary_stab     >= th_primary_st) &
  (h2_med           >= th_h2_keep)

tentative <- (!keep) & (
  (med_abs_primary >= th_load_med & salience_prob >= 0.60) |
    (h2_med >= 0.15 & salience_prob >= 0.60) |
    (primary_stab >= 0.60 & med_abs_primary >= 0.25)
)

drop <- !(keep | tentative)

# 12.9 Optional MSA backstop
if (has_MSA) {
  keep      <- keep      & (MSA_i[vars] >= th_msa)
  tentative <- tentative & (MSA_i[vars] >= th_msa)
  drop      <- drop | (MSA_i[vars] < th_msa)
}

# 12.10 Announce
msg_items <- function(ix) if (any(ix)) paste(names(ix)[ix], collapse = ", ") else "(none)"
message("KEEP: ",      msg_items(keep))
message("TENTATIVE: ", msg_items(tentative))
message("DROP: ",      msg_items(drop))

# 12.11 Build pruned Λ, Ψ, R (KEEP + TENTATIVE retained, DROP removed)
keep_or_tent <- names(keep)[keep | tentative]
Lambda0      <- L_median[keep_or_tent, , drop = FALSE]
Psi0         <- psi_median[keep_or_tent]
R_prune      <- R[keep_or_tent, keep_or_tent, drop = FALSE]  # Spearman-only

# 12.12 Iterative post-prune refit and oblique communality backstop
repeat {
  fa_pruned <- psych::fa(R_prune, nfactors = ncol(Lambda0), fm = "minres",
                         rotate = "geominQ", n.obs = nrow(df_num))
  
  L_new   <- as.matrix(fa_pruned$loadings)
  Phi_new <- fa_pruned$Phi
  h2_new  <- rowSums((L_new %*% Phi_new) * L_new)  # oblique h^2
  
  drop_comm <- names(h2_new)[h2_new < th_h2_drop]
  if (length(drop_comm) == 0) break
  
  message("Dropping for very low communality (< ", th_h2_drop, "): ",
          paste(drop_comm, collapse = ", "))
  
  keep_final <- setdiff(rownames(R_prune), drop_comm)
  # Update to the new, smaller variable set
  R_prune <- R_prune[keep_final, keep_final, drop = FALSE]
  Lambda0 <- Lambda0[keep_final, , drop = FALSE]
  Psi0    <- Psi0[keep_final]
}
# If nothing was dropped in the loop, define keep_final now
if (!exists("keep_final")) keep_final <- rownames(R_prune)

# ---------------------------
# Step 12b. φ bootstrap only
# ---------------------------
B <- 1000
k <- ncol(Lambda0)

n_cores <- max(1, parallel::detectCores() - 1)
cl      <- parallel::makeCluster(n_cores)
doSNOW::registerDoSNOW(cl)

pb       <- txtProgressBar(max = B, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts     <- list(progress = progress)

phis_res <- foreach::foreach(
  b = 1:B, .combine = rbind,
  .packages = c("psych","Matrix","clue"),
  .export   = c("align_to_ref","Lambda0","k","keep_final", "df_num")
  , .options.snow = opts
) %dopar% {
  repeat {
    # a) bootstrap sample on retained variables
    samp_idx <- sample(nrow(df_num), replace = TRUE)
    samp     <- df_num[samp_idx, keep_final, drop = FALSE]
    
    # b) Spearman correlation and PD repair
    Rb <- tryCatch(
      cor(as.data.frame(lapply(samp, as.numeric)),
          method = "spearman", use = "pairwise.complete.obs"),
      error = function(e) NULL
    )
    if (is.null(Rb) || any(is.na(Rb))) next
    Rb <- as.matrix(Matrix::nearPD(Rb, corr = TRUE, keepDiag = TRUE)$mat)
    
    # c) EFA (MINRES + geominQ)
    fa_b <- tryCatch(
      psych::fa(Rb, nfactors = k, fm = "minres",
                rotate = "geominQ", n.obs = nrow(samp)),
      error = function(e) NULL
    )
    if (is.null(fa_b)) next
    
    # d) Align bootstrap loadings to the pruned target Lambda0
    Lb <- align_to_ref(fa_b$loadings, Lambda0)$L
    
    # e) Tucker's congruence (φ) with target
    phi_b <- diag(psych::factor.congruence(Lambda0, Lb))
    
    return(phi_b)   # φ only
  }
}

close(pb); parallel::stopCluster(cl)

# Summaries for φ
phis_rob  <- as.matrix(phis_res)          # B x k
colnames(phis_rob) <- colnames(Lambda0)

phi_mean  <- colMeans(phis_rob, na.rm = TRUE)
phi_med   <- apply(phis_rob, 2, median,  na.rm = TRUE)
phi_ci    <- apply(phis_rob, 2, quantile, probs = c(.025, .975), na.rm = TRUE)

cat(sprintf("Finished %d valid φ bootstraps\n", nrow(phis_rob)))
cat("Mean φ by factor:   ", round(phi_mean, 3), "\n")
cat("Median φ by factor: ", round(phi_med, 3),  "\n")
print(t(round(phi_ci, 3)))


# PROBABLY CFA

# ---------------------------------------------------------------------------
# Step 15. Communalities & Residual Diagnostics (oblique, from final fit)
# ---------------------------------------------------------------------------

# At this point, fa_pruned / L_new / Phi_new were produced by the final
# iteration of Step 12.12 and are guaranteed to match R_prune’s variables.

# Oblique communalities: h^2 = λ' Φ λ
h2 <- rowSums((L_new %*% Phi_new) * L_new)
cat("Mean communality (h²):", round(mean(h2, na.rm = TRUE), 3), "\n")
print(head(data.frame(variable = rownames(L_new), communality = h2), 10))

# Uniquenesses and Psi
Psi_vec <- fa_pruned$uniquenesses
Psi_mat <- diag(Psi_vec)
rownames(Psi_mat) <- names(Psi_vec)
colnames(Psi_mat) <- names(Psi_vec)

# Model-implied correlation: Σ_hat = Λ Φ Λ' + Ψ
Sigma_hat <- L_new %*% Phi_new %*% t(L_new) + Psi_mat

# --- Safety alignment in case column orders drifted upstream ---
common <- intersect(rownames(R_prune), rownames(Sigma_hat))
if (length(common) < nrow(R_prune)) {
  warning("Aligning Sigma_hat to R_prune by common variables: ",
          paste(setdiff(rownames(R_prune), common), collapse = ", "))
}
R_prune   <- R_prune[common, common, drop = FALSE]
Sigma_hat <- Sigma_hat[common, common, drop = FALSE]
# ----------------------------------------------------------------

# Residuals and RMSR
resid_mat       <- R_prune - Sigma_hat
diag(resid_mat) <- 0

off_diag_vals <- resid_mat[lower.tri(resid_mat, diag = FALSE)]
RMSR <- sqrt(mean(off_diag_vals^2, na.rm = TRUE))
cat("RMSR (off-diagonal) =", round(RMSR, 4), "\n")

# Flag residuals with |residual| > 0.10
thr_resid <- 0.10
off_idx <- which(abs(resid_mat) > thr_resid & row(resid_mat) > col(resid_mat), arr.ind = TRUE)
if (nrow(off_idx) > 0) {
  offenders <- data.frame(
    var1     = rownames(resid_mat)[off_idx[,1]],
    var2     = colnames(resid_mat)[off_idx[,2]],
    residual = resid_mat[off_idx]
  )
  cat("Residuals exceeding |", thr_resid, "|:\n", sep = "")
  print(offenders)
} else {
  cat("No residuals exceed |", thr_resid, "|.\n", sep = "")
}

# Heatmap (oblique residuals)
df_long <- reshape2::melt(resid_mat, varnames = c("Row", "Col"),
                          value.name = "Residual")
ggplot2::ggplot(df_long, ggplot2::aes(x = Col, y = Row, fill = Residual)) +
  ggplot2::geom_tile() +
  ggplot2::scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
  ggplot2::geom_text(ggplot2::aes(label = round(Residual, 2)), size = 2.5) +
  ggplot2::theme_minimal() +
  ggplot2::theme(
    axis.text.x = ggplot2::element_text(angle = 45, hjust = 1, size = 8),
    axis.text.y = ggplot2::element_text(size = 8),
    panel.grid  = ggplot2::element_blank()
  ) +
  ggplot2::labs(fill = "Residual")

# ---------------------------------------------------------------------------
# 16. Restrict summaries to final kept items + assemble a consolidated EFA report
# ---------------------------------------------------------------------------

# a) Filter CIs to the final kept variables
df_L_ci_pruned <- dplyr::filter(df_L_ci, variable %in% keep_final)

# b) Pruned median loading matrix (aligned medians from bootstrap)
L_median_pruned <- L_median[keep_final, , drop = FALSE]

# c) Map each variable to its primary factor index (j_star) and names
# j_star is length p (over 'vars'); align to keep_final
j_map  <- setNames(j_star, vars)      # index 1..k for all vars
j_keep <- as.integer(j_map[keep_final])
# Use the ACTUAL column names of L_median_pruned to avoid "F1" vs "MR1" mismatches
L_median_pruned <- L_median[keep_final, , drop = FALSE]
pf_keep <- colnames(L_median_pruned)[j_keep]   # e.g., "F1","F2",...

# d) Pull the primary-factor CI row for each kept variable
df_L_ci_pruned <- dplyr::filter(df_L_ci, variable %in% keep_final)

# Sanity checks to catch naming misalignments early
stopifnot(all(pf_keep %in% colnames(L_median_pruned)))
stopifnot(all(df_L_ci_pruned$factor %in% colnames(L_median_pruned)))

primary_rows <- do.call(
  rbind,
  lapply(seq_along(keep_final), function(i) {
    v  <- keep_final[i]
    jf <- pf_keep[i]                                # factor name (matches columns)
    # median loading (signed) on primary from L_median_pruned
    med <- L_median_pruned[v, jf, drop = TRUE]
    # CI for primary factor from df_L_ci_pruned
    ci_row <- df_L_ci_pruned[df_L_ci_pruned$variable == v & df_L_ci_pruned$factor == jf, ]
    if (nrow(ci_row) == 0L) ci_row <- data.frame(lower = NA_real_, upper = NA_real_)
    data.frame(
      variable = v,
      primary_factor = jf,
      median_loading = as.numeric(med),
      lower = ci_row$lower, upper = ci_row$upper,
      stringsAsFactors = FALSE
    )
  })
)


# e) Bring in stability & communality diagnostics from Step 12 + final refit
#    - salience_prob, primary_stab, h2_med were named by variable in Step 12
sal_keep  <- salience_prob[keep_final]
pst_keep  <- primary_stab[keep_final]
h2_med_k  <- h2_med[keep_final]                 # orthogonal proxy (bootstrap median)
h2_obl_k  <- h2_new[keep_final]                 # final oblique h^2 from fa_pruned

# f) Consolidated EFA report table
efa_report <- dplyr::as_tibble(primary_rows) |>
  dplyr::mutate(
    salience_prob   = as.numeric(sal_keep[variable]),
    primary_stab    = as.numeric(pst_keep[variable]),
    h2_med_boot     = as.numeric(h2_med_k[variable]),
    h2_oblique_final= as.numeric(h2_obl_k[variable]),
    ci_cross_zero   = (lower <= 0 & upper >= 0),
    abs_median      = abs(median_loading)
  ) |>
  dplyr::arrange(primary_factor, dplyr::desc(abs_median))

# g) Print a compact report and overall factor-level φ summary (from Step 12b)
cat("\n=== Consolidated EFA Report (final kept items) ===\n")
print(dplyr::select(
  efa_report, variable, primary_factor, median_loading, lower, upper,
  salience_prob, primary_stab, h2_med_boot, h2_oblique_final, ci_cross_zero
))

cat("\n=== Factor-level congruence (φ) summary ===\n")
cat("Mean φ by factor:   ", paste(names(phi_mean), round(phi_mean, 3), sep="=", collapse="  "), "\n")
cat("Median φ by factor: ", paste(names(phi_med),  round(phi_med,  3), sep="=", collapse="  "), "\n")
print(t(round(phi_ci, 3)))

# ---------------------------------------------------------------------------
# 17. Stability & cross-loading summaries (pruned set)
# ---------------------------------------------------------------------------

# 17a) Unstable primary loadings = primary CI crosses zero
unstable_primary <- dplyr::filter(efa_report, ci_cross_zero)
cat("\n=== Unstable primary loadings (primary CI spans 0) ===\n")
if (nrow(unstable_primary)) print(unstable_primary[, c("variable","primary_factor","median_loading","lower","upper")]) else cat("(none)\n")

# 17b) Cross-loading candidates = variables with ≥2 factors whose CI excludes 0
cross_pruned <- df_L_ci_pruned |>
  dplyr::mutate(nonzero = (lower > 0 | upper < 0)) |>
  dplyr::filter(nonzero) |>
  dplyr::group_by(variable) |>
  dplyr::summarise(
    n_factors = dplyr::n(),
    intervals = paste0(factor, "[", round(lower, 2), ",", round(upper, 2), "]", collapse = "; "),
    .groups = "drop"
  ) |>
  dplyr::filter(n_factors > 1)

cat("\n=== Cross-loading candidates (≥ 2 non-zero CIs) ===\n")
if (nrow(cross_pruned)) print(cross_pruned) else cat("(none)\n")

# 17c) Strong labeling candidates = primary CI entirely beyond ±0.30
label_pruned <- efa_report |>
  dplyr::filter(lower >= 0.30 | upper <= -0.30) |>
  dplyr::arrange(primary_factor, dplyr::desc(abs_median))

cat("\n=== Labeling candidates (primary CI beyond ±0.30) ===\n")
if (nrow(label_pruned)) print(label_pruned[, c("variable","primary_factor","median_loading","lower","upper")]) else cat("(none)\n")

# ---------------------------------------------------------------------------
# 18. CI errorbar plot (primary factor only, pruned set)
# ---------------------------------------------------------------------------

# Order variables: by primary factor, then by |median loading|
efa_report$var_order <- seq_len(nrow(efa_report))
efa_report <- efa_report |>
  dplyr::arrange(primary_factor, dplyr::desc(abs_median)) |>
  dplyr::mutate(variable_f = factor(variable, levels = variable))

ggplot2::ggplot(
  efa_report,
  ggplot2::aes(x = variable_f, y = median_loading, ymin = lower, ymax = upper, colour = primary_factor)
) +
  ggplot2::geom_errorbar(width = 0.2, position = ggplot2::position_dodge(width = 0.6)) +
  ggplot2::geom_point(size = 2, position = ggplot2::position_dodge(width = 0.6)) +
  ggplot2::geom_hline(yintercept = 0, linetype = 2) +
  ggplot2::coord_flip() +
  ggplot2::labs(
    x = NULL,
    y = "Primary median loading ±95% CI",
    title = "Pruned Items: Primary Factor Loadings with 95% Bootstrap CIs"
  ) +
  ggplot2::theme_minimal() +
  ggplot2::theme(legend.position = "bottom")

# ---------------------------------------------------------------------------
# 19. Heatmap of pruned median loadings (grouped by primary factor)
# ---------------------------------------------------------------------------

# Long format for heatmap
L_df_pruned <- as.data.frame(L_median_pruned)
L_df_pruned$variable <- rownames(L_df_pruned)
L_long_pruned <- tidyr::pivot_longer(L_df_pruned, -variable, names_to = "factor", values_to = "loading")

# Add primary factor & ordering to group rows in the heatmap
pf_df <- efa_report[, c("variable","primary_factor","abs_median")]
L_long_pruned <- dplyr::left_join(L_long_pruned, pf_df, by = "variable")

# Order variables within primary factor by |median loading|
var_levels <- efa_report$variable  # already ordered by factor then |loading|
L_long_pruned$variable_f <- factor(L_long_pruned$variable, levels = var_levels)

ggplot2::ggplot(L_long_pruned, ggplot2::aes(x = factor, y = variable_f, fill = loading)) +
  ggplot2::geom_tile() +
  ggplot2::scale_fill_gradient2(
    low = "blue", mid = "white", high = "red", midpoint = 0, name = "Loading"
  ) +
  ggplot2::labs(
    title = "Pruned: Median Factor Loadings (aligned, bootstrap medians)",
    x = NULL, y = NULL
  ) +
  ggplot2::theme_minimal() +
  ggplot2::theme(
    axis.text.x    = ggplot2::element_text(angle = 45, hjust = 1),
    axis.text.y    = ggplot2::element_text(size = 8),
    legend.position = "right"
  )

# =============================================================================
# 20. CFA
# =============================================================================

## 1. Determine the variable set ----

loading_matrix <- L_new

factor_names <- colnames(loading_matrix)
if (is.null(factor_names) || length(factor_names) == 0) {
  factor_names <- paste0("F", seq_len(ncol(loading_matrix)))
  colnames(loading_matrix) <- factor_names
}
cfa_vars <- rownames(loading_matrix)
if (is.null(cfa_vars) || length(cfa_vars) == 0) {
  stop("Loading matrix must have row names corresponding to variables.")
}

## 2. Extract the relevant data and infer variable types ----
# Extract only the variables used in the CFA from the full mixed dataset.  A
# safety check ensures all are present.
missing_vars <- setdiff(cfa_vars, names(df_num))
if (length(missing_vars) > 0) {
  stop("The following variables required by the CFA are missing from df_num: ",
       paste(missing_vars, collapse = ", "))
}
cfa_df <- df_num[, cfa_vars, drop = FALSE]

# Determine which variables are continuous and which are ordered.  For
# continuous variables we rely on R's class (numeric/integer); for ordered
# variables we check if they are factors or ordered factors.  The user may
# override these defaults by adjusting the lists before fitting.
is_num    <- sapply(cfa_df, is.numeric)
is_ord    <- sapply(cfa_df, function(x) inherits(x, c("factor", "ordered")))
ordered_items <- names(cfa_df)[is_ord]
cont_vars     <- names(cfa_df)[is_num]

message("[CFA] Variable classes:")
print(sapply(cfa_df, class))
message("[CFA] Continuous variables:")
print(cont_vars)
message("[CFA] Ordered/ordinal variables:")
print(ordered_items)

# Check for constant variables (all values identical or all NA) and warn.
for (v in names(cfa_df)) {
  vals <- na.omit(cfa_df[[v]])
  if (length(unique(vals)) <= 1) {
    warning(sprintf("Variable %s is constant or has no variability.", v))
  }
}

# Check for near perfect correlations among continuous variables and warn.
if (length(cont_vars) > 1) {
  cor_mat <- suppressWarnings(stats::cor(cfa_df[cont_vars], use = "pairwise.complete.obs"))
  high_corrs <- which(abs(cor_mat) > 0.95 & abs(cor_mat) < 1, arr.ind = TRUE)
  if (nrow(high_corrs) > 0) {
    message("[CFA WARNING] High correlations (>0.95) detected among continuous variables:")
    for (k in seq_len(nrow(high_corrs))) {
      i1 <- rownames(cor_mat)[high_corrs[k, 1]]
      i2 <- colnames(cor_mat)[high_corrs[k, 2]]
      message(sprintf("  • %s and %s: %.3f", i1, i2, cor_mat[high_corrs[k, 1], high_corrs[k, 2]]))
    }
  }
}

## 3. Preprocessing: scale continuous variables and coerce ordered ----

# helper: coerce any vector to numeric safely
.to_numeric <- function(x) {
  if (is.numeric(x)) return(x)
  if (is.factor(x) || is.ordered(x)) return(as.numeric(x))            # factor codes
  if (is.logical(x)) return(as.numeric(x))
  # last resort: try character -> numeric
  suppressWarnings(as.numeric(as.character(x)))
}

if (length(cont_vars) > 0) {
  # 1) make a numeric data.frame
  num_df <- as.data.frame(lapply(cfa_df[cont_vars], .to_numeric), 
                          stringsAsFactors = FALSE)
  
  # 2) mean-center everything
  means <- sapply(num_df, function(v) mean(v, na.rm = TRUE))
  centered <- sweep(num_df, 2, means, FUN = "-")
  
  # 3) scale by SD where possible (avoid divide-by-zero for constant cols)
  sds <- sapply(num_df, function(v) sd(v, na.rm = TRUE))
  ok  <- is.finite(sds) & sds > .Machine$double.eps
  
  scaled <- centered
  if (any(ok)) {
    scaled[, ok] <- sweep(centered[, ok, drop = FALSE], 2, sds[ok], FUN = "/")
  }
  # columns with zero variance are left mean-centered only
  
  # 4) put back
  colnames(scaled) <- cont_vars
  cfa_df[cont_vars] <- scaled
}

# make ordered variables truly ordered (do NOT include them in cont_vars)
if (length(ordered_items) > 0) {
  for (v in ordered_items) {
    if (!is.ordered(cfa_df[[v]])) {
      cfa_df[[v]] <- ordered(cfa_df[[v]], levels = sort(unique(cfa_df[[v]])))
    }
  }
}

# --- Helpers used later (safe CFA fit, compact fit print, sparse collapse) ---
if (!exists("safe_cfa")) {
  safe_cfa <- function(model_text, data, ordered_vars = NULL) {
    est <- if (!is.null(ordered_vars) && length(ordered_vars) > 0) "WLSMV" else "MLR"
    ord <- if (!is.null(ordered_vars) && length(ordered_vars) > 0) ordered_vars else NULL
    tryCatch(
      lavaan::cfa(model = model_text, data = data, std.lv = TRUE, estimator = est, ordered = ord),
      error = function(e) { message("[safe_cfa] ", e$message); return(NULL) }
    )
  }
}

if (!exists("fit_print")) {
  fit_print <- function(fit, tag = "[CFA]") {
    if (is.null(fit)) { message(tag, " fit is NULL"); return(invisible(NULL)) }
    fm <- lavaan::fitMeasures(fit, c("chisq","df","pvalue","cfi","tli","rmsea","srmr"))
    cat(sprintf("%s χ²(%d)=%.2f, p=%.3f; CFI=%.3f, TLI=%.3f, RMSEA=%.3f, SRMR=%.3f\n",
                tag, as.integer(fm["df"]), fm["chisq"], fm["pvalue"],
                fm["cfi"], fm["tli"], fm["rmsea"], fm["srmr"]))
  }
}

if (!exists("collapse_sparse_categories")) {
  collapse_sparse_categories <- function(data, var_name, min_count = 5) {
    if (!var_name %in% names(data)) return(data)
    x <- data[[var_name]]
    if (!is.factor(x) && !is.ordered(x)) return(data)
    tab <- table(x, useNA = "no")
    sparse <- names(tab)[tab < min_count]
    if (!length(sparse)) return(data)
    y <- as.character(x)
    # simple adjacent collapsing for ordered; "Other" for nominal
    if (is.ordered(x)) {
      lev <- levels(x)
      for (lv in sparse) {
        pos <- match(lv, lev)
        repl <- if (!is.na(pos) && pos > 1) lev[pos - 1] else if (!is.na(pos) && pos < length(lev)) lev[pos + 1] else lv
        y[y == lv] <- repl
      }
      data[[var_name]] <- ordered(y, levels = unique(y))
    } else {
      y[y %in% sparse] <- "Other"
      data[[var_name]] <- factor(y)
    }
    data
  }
}

## 4. Construct the CFA model syntax from the EFA loadings ----
# By default: one primary factor per item (simple structure).
# Optionally allow cross-loadings above 'cfa_cross_threshold' (e.g., 0.30).

cfa_cross_threshold <- get0("cfa_cross_threshold", ifnotfound = NA_real_) # set to 0.30 if you want
Lmat <- as.matrix(loading_matrix)
rownames(Lmat) <- cfa_vars
colnames(Lmat) <- factor_names

# Primary factor = argmax |loading| for each item
j_primary <- apply(abs(Lmat), 1, which.max)  # integer in 1..k
tab_primary <- table(factor(j_primary, levels = seq_len(ncol(Lmat))))
if (any(tab_primary < 2)) {
  message("[CFA WARNING] Some factors have < 2 primary indicators: ",
          paste(factor_names[which(tab_primary < 2)], collapse = ", "),
          ". Identification may be weak.")
}

# Build measurement lines
cfa_model_lines <- character(0)
for (j in seq_len(ncol(Lmat))) {
  # primary items for factor j
  items_j <- cfa_vars[j_primary == j]
  
  # optional cross-loadings above threshold (exclude primary assignments)
  if (is.finite(cfa_cross_threshold) && cfa_cross_threshold > 0) {
    cross_idx <- which(abs(Lmat[, j]) >= cfa_cross_threshold & j_primary != j)
    items_j <- unique(c(items_j, cfa_vars[cross_idx]))
  }
  
  if (length(items_j) > 0) {
    cfa_model_lines <- c(
      cfa_model_lines,
      paste0(factor_names[j], " =~ ", paste(items_j, collapse = " + "))
    )
  }
}

# Factor covariances (all freely correlated)
if (ncol(Lmat) > 1) {
  cov_lines <- apply(utils::combn(factor_names, 2), 2,
                     function(x) paste(x[1], "~~", x[2]))
  cfa_model_lines <- c(cfa_model_lines, cov_lines)
}

cfa_model_text <- paste(cfa_model_lines, collapse = "\n")
message("\n[CFA] Fitting CFA with the following syntax",
        if (is.finite(cfa_cross_threshold)) paste0(" (cross_thresh=", cfa_cross_threshold, ")") else " (simple structure)",
        ":\n", cfa_model_text, "\n")


## 5. Fit the CFA model using lavaan (auto estimator) ----
estimator_auto <- if (length(ordered_items) > 0) "WLSMV" else "MLR"
ordered_arg    <- if (length(ordered_items) > 0) ordered_items else NULL

fit_cfa <- tryCatch({
  lavaan::cfa(
    model     = cfa_model_text,
    data      = cfa_df,
    std.lv    = TRUE,
    estimator = estimator_auto,
    ordered   = ordered_arg
  )
}, error = function(e) {
  message("[CFA ERROR] Could not fit the CFA model: ", e$message)
  return(NULL)
})

if (is.null(fit_cfa)) stop("Initial CFA model fit failed.")


## 6. Basic diagnostics: fit measures and loadings ----
message("[CFA] Initial fit indices:")
print(lavaan::fitMeasures(fit_cfa, c("chisq","df","pvalue","cfi","tli","rmsea","srmr")))
message("[CFA] Standardized loadings (|λ| > 0.30):")
std_loads <- lavaan::inspect(fit_cfa, "std")$lambda
if (!is.null(std_loads)) {
  for (f in colnames(std_loads)) {
    significant <- abs(std_loads[, f]) > 0.30
    if (any(significant)) {
      cat(sprintf("  Factor %s:\n", f))
      load_vals <- std_loads[significant, f]
      for (nm in names(load_vals)) {
        cat(sprintf("    %s : %.3f\n", nm, load_vals[nm]))
      }
    }
  }
}

## 7. Modification indices: identify areas for improvement ----
mi_tbl <- tryCatch(lavaan::modindices(fit_cfa),
                   error = function(e) { message("[CFA] MI unavailable: ", e$message); NULL })

mi_thresh <- 10
if (!is.null(mi_tbl)) {
  mi_tbl <- mi_tbl[order(-mi_tbl$mi), ]
  high_mi <- subset(mi_tbl, mi > mi_thresh & !(op %in% c("~1","~")))
  if (nrow(high_mi) > 0) {
    message("[CFA] Modification indices (MI > ", mi_thresh, "):")
    print(high_mi[, c("lhs","op","rhs","mi")])
  } else {
    message("[CFA] No modification indices greater than ", mi_thresh, " found.")
  }
} else {
  message("[CFA] Skipping MI listing (information matrix was singular).")
}


## 8. Optional: automated MI refinement (up to max_steps modifications) ----
max_steps <- 5
cfa_refinement <- function(model_text, fit, data, ordered_vars, steps) {
  current_model <- model_text
  current_fit   <- fit
  added         <- character(0)
  for (s in seq_len(steps)) {
    mi <- tryCatch(lavaan::modindices(current_fit), error = function(e) NULL)
    if (is.null(mi)) break
    mi <- mi[order(-mi$mi), ]
    mi <- subset(mi, mi > mi_thresh & !(op %in% c("~1","~")))
    mi <- mi[!paste(mi$lhs, mi$op, mi$rhs) %in% added, ]
    if (nrow(mi) == 0) break
    top <- mi[1, ]
    new_term <- paste(top$lhs, top$op, top$rhs)
    current_model <- paste(current_model, new_term, sep="\n")
    added <- c(added, new_term)
    message(sprintf("[MI] Adding: %s (MI = %.2f)\n", new_term, top$mi))
    current_fit <- tryCatch({
      lavaan::cfa(model = current_model, data = data, std.lv = TRUE,
                  estimator = "WLSMV", ordered = ordered_vars)
    }, error = function(e) { message("[MI ERROR] ", e$message); return(current_fit) })
  }
  list(model = current_model, fit = current_fit, added = added)
}

mi_results <- cfa_refinement(cfa_model_text, fit_cfa, cfa_df, ordered_items, max_steps)
cfa_model_refined <- mi_results$model
fit_cfa_refined  <- mi_results$fit

message("\n[CFA] Refined model fit indices:")
print(lavaan::fitMeasures(fit_cfa_refined, c("chisq","df","pvalue","cfi","tli","rmsea","srmr")))

## 9. Save results, factor scores, residual plots (optional) ----
saveRDS(fit_cfa_refined, file = "fit_cfa_refined.rds")
if (requireNamespace("semPlot", quietly = TRUE)) {
  pdf("cfa_refined_diagram.pdf", width = 10, height = 8)
  semPlot::semPaths(fit_cfa_refined, "std", whatLabels = "std", edge.label.cex = 1.1, layout = "tree", style = "lisrel")
  dev.off()
}
factor_scores <- tryCatch(lavaan::lavPredict(fit_cfa_refined, method="Bartlett"), error = function(e) NULL)
if (!is.null(factor_scores)) {
  utils::write.csv(factor_scores, "factor_scores.csv", row.names = TRUE)
}

resid_mat_cfa <- tryCatch(lavaan::residuals(fit_cfa_refined, type="cor")$cov, error = function(e) NULL)
if (!is.null(resid_mat_cfa)) {
  stats::heatmap(resid_mat_cfa, main = "Refined CFA Residual Correlation Matrix", symm = TRUE)
}

## 10. Advanced diagnostics: negative residual variances, alternative factor structures ----
# Identify Heywood cases: parameters where a variable's residual variance is negative
par_table_ref <- lavaan::parTable(fit_cfa_refined)
neg_resid <- subset(par_table_ref, op == "~~" & lhs == rhs & est < 0)
if (nrow(neg_resid) > 0) {
  message("\n[CFA] Heywood cases detected (negative residual variances):")
  print(neg_resid[, c("lhs", "est")])
} else {
  message("\n[CFA] No Heywood cases detected.")
}

# Simple example of probing an alternative factor structure: test a bifactor model if
# there are at least three factors.  This is provided as a template and may
# require adjustment for specific datasets.
if (ncol(loading_matrix) >= 3) {
  message("\n[CFA] Testing bifactor model (general + specific factors)...")
  # General factor loads on all items; specific factors defined per original
  gen_line <- paste("G =~", paste(cfa_vars, collapse = " + "))
  spec_lines <- sapply(seq_len(ncol(loading_matrix)), function(j) {
    # Items with highest loading on factor j
    items <- cfa_vars[abs(loading_matrix[, j]) > 0]
    paste0("S", j, " =~ ", paste(items, collapse=" + "))
  })
  bf_model <- paste(c(gen_line, spec_lines, paste0("G ~~ 0*", paste(paste0("S", seq_len(ncol(loading_matrix))), collapse = " + "))), collapse = "\n")
  fit_bf <- safe_cfa(bf_model, cfa_df, ordered_items)
  if (!is.null(fit_bf)) {
    fit_print(fit_bf, "[BIFACTOR]")
  }
}

## 11. Split‑half cross‑validation ----
if (nrow(cfa_df) > 1) {
  set.seed(2025)
  idx <- sample(nrow(cfa_df))
  half <- floor(length(idx) / 2)
  split1 <- cfa_df[idx[1:half], , drop = FALSE]
  split2 <- cfa_df[idx[(half+1):length(idx)], , drop = FALSE]
  fit_s1 <- safe_cfa(cfa_model_refined, split1, ordered_items)
  fit_s2 <- safe_cfa(cfa_model_refined, split2, ordered_items)
  if (!is.null(fit_s1) && !is.null(fit_s2)) {
    L1 <- lavaan::inspect(fit_s1, "std")$lambda
    L2 <- lavaan::inspect(fit_s2, "std")$lambda
    common_items <- intersect(rownames(L1), rownames(L2))
    if (length(common_items) > 0) {
      diff_abs <- abs(L1[common_items, ] - L2[common_items, ])
      message("\n[CFA] Split‑half cross‑validation: mean |Δ loading| = ", round(mean(diff_abs), 3), "; max |Δ| = ", round(max(diff_abs), 3))
    }
  }
}

## 12. Bootstrap cross‑validation for loadings (if parallel backend available) ----
if (requireNamespace("foreach", quietly = TRUE) &&
    requireNamespace("doSNOW", quietly = TRUE) &&
    requireNamespace("parallel", quietly = TRUE)) {
  B_boot <- 100
  n_cores <- max(1, parallel::detectCores() - 1)
  cl_boot <- parallel::makeCluster(n_cores)
  doSNOW::registerDoSNOW(cl_boot)
  pb_boot <- txtProgressBar(max = B_boot, style = 3)
  opts_boot <- list(progress = function(n) setTxtProgressBar(pb_boot, n))
  items_order <- rownames(lavaan::inspect(fit_cfa_refined, "std")$lambda)
  boot_load <- foreach::foreach(b = 1:B_boot, .combine = rbind,
                                .packages = "lavaan", .options.snow = opts_boot) %dopar% {
                                  samp_idx <- sample(nrow(cfa_df), replace = TRUE)
                                  dat_b <- cfa_df[samp_idx, ]
                                  fit_b <- tryCatch(lavaan::cfa(cfa_model_refined, data = dat_b, std.lv = TRUE,
                                                                estimator = "WLSMV", ordered = ordered_items), error = function(e) NULL)
                                  if (is.null(fit_b)) return(rep(NA, length(items_order) * ncol(loading_matrix)))
                                  as.vector(lavaan::inspect(fit_b, "std")$lambda[items_order, , drop = FALSE])
                                }
  close(pb_boot)
  parallel::stopCluster(cl_boot)
  boot_conv <- mean(!is.na(boot_load[, 1]))
  message("[CFA] Bootstrap convergence rate: ", round(boot_conv * 100, 1), "%")
  if (boot_conv >= 0.95) {
    boot_ci <- apply(boot_load, 2, stats::quantile, c(0.025, 0.975), na.rm = TRUE)
    message("[CFA] Bootstrap 95% confidence intervals computed.")
  } else {
    message("[CFA] Bootstrap convergence <95%; results may be unreliable.")
  }
}

## 13. Collapsing sparse categories for ordered variables ----
# Optionally collapse sparse categories (<5 observations) to improve model fit.
collapse_sparse <- TRUE
if (collapse_sparse) {
  cfa_df_collapsed <- cfa_df
  for (v in ordered_items) {
    cfa_df_collapsed <- collapse_sparse_categories(cfa_df_collapsed, v, min_count = 5)
  }
  ordered_items_collapsed <- names(Filter(is.ordered, cfa_df_collapsed))
  fit_cfa_collapsed <- safe_cfa(cfa_model_refined, cfa_df_collapsed, ordered_items_collapsed)
  if (!is.null(fit_cfa_collapsed)) {
    message("\n[CFA] Collapsed categories model fit indices:")
    print(lavaan::fitMeasures(fit_cfa_collapsed, c("chisq","df","pvalue","cfi","tli","rmsea","srmr")))
  }
}

## 14. Final model selection and summary ----
# Choose the best among original, refined, and collapsed models by CFI (highest)
# and RMSEA (lowest).  All models are retained for inspection; best is printed.
model_candidates <- list(original = fit_cfa, refined = fit_cfa_refined)
if (exists("fit_cfa_collapsed") && !is.null(fit_cfa_collapsed)) {
  model_candidates$collapsed <- fit_cfa_collapsed
}
model_metrics <- sapply(model_candidates, function(fit) {
  fm <- lavaan::fitMeasures(fit, c("cfi", "rmsea", "srmr"))
  c(CFI = fm["cfi"], RMSEA = fm["rmsea"], SRMR = fm["srmr"])
})
# Highest CFI and lowest RMSEA wins; tie broken by SRMR
order_indices <- order(-model_metrics["CFI", ], model_metrics["RMSEA", ], model_metrics["SRMR", ])
best_name <- names(model_candidates)[order_indices[1]]
best_fit  <- model_candidates[[best_name]]
message("\n[CFA] Best final model: ", best_name)
print(model_metrics[, best_name])

## 15. Final measurement model report (delta parameterisation) ----
# Build a concise report for the best model.  Use factor loadings, fit indices,
# factor correlations, and a composite reliability estimate for each factor.
Lam_best <- lavaan::inspect(best_fit, "std")$lambda
Psi_best <- lavaan::inspect(best_fit, "std")$psi
fm_best  <- lavaan::fitMeasures(best_fit, c("chisq","df","pvalue","cfi","tli","rmsea","rmsea.ci.lower","rmsea.ci.upper","srmr"))

message("\n[FINAL MODEL REPORT]\n")
cat(sprintf("χ²(%d) = %.2f, p = %.3f\n", as.integer(fm_best["df"]), fm_best["chisq"], fm_best["pvalue"]))
cat(sprintf("CFI = %.3f, TLI = %.3f, RMSEA = %.3f [%.3f, %.3f], SRMR = %.3f\n",
            fm_best["cfi"], fm_best["tli"], fm_best["rmsea"], fm_best["rmsea.ci.lower"],
            fm_best["rmsea.ci.upper"], fm_best["srmr"]))

# Report loadings greater than .30
for (f in colnames(Lam_best)) {
  lam_f <- Lam_best[, f]
  sel <- abs(lam_f) > 0.30
  if (any(sel)) {
    cat(sprintf("Factor %s loadings (|λ| > .30):\n", f))
    for (nm in names(lam_f[sel])) {
      cat(sprintf("  • %s : %.3f\n", nm, lam_f[nm]))
    }
  }
}

# Factor correlations
if (!is.null(Psi_best) && ncol(Psi_best) > 1) {
  for (i in 1:(ncol(Psi_best)-1)) {
    for (j in (i+1):ncol(Psi_best)) {
      r12 <- Psi_best[i, j]
      cat(sprintf("Factor correlation (%s ↔ %s): %.3f\n", colnames(Psi_best)[i], colnames(Psi_best)[j], r12))
    }
  }
}

# Composite reliability estimate per factor
composite_rel <- function(loads) {
  loads <- loads[abs(loads) > 0.30]
  if (length(loads) < 2) return(NA_real_)
  sl  <- sum(loads)
  sl2 <- sum(loads^2)
  se  <- length(loads) - sl2
  (sl^2) / ((sl^2) + se)
}
for (f in colnames(Lam_best)) {
  cr <- composite_rel(Lam_best[, f])
  cat(sprintf("Composite reliability (%s): %s\n", f, ifelse(is.na(cr), "NA", sprintf("%.3f", cr))))
}

cat("\n[NOTE] The final model uses WLSMV (delta parameterisation). Ordered variables are automatically treated as ordinal, and continuous variables are scaled prior to fitting.\n")




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

