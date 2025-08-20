
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

lambda_boot <- boot_load[, 1:(p*k), drop = FALSE]
psi_boot    <- boot_load[, (p*k+1):(p*k+p), drop = FALSE]

L_median <- matrix(
  apply(lambda_boot, 2, median),
  nrow = p, ncol = k, byrow = FALSE,
  dimnames = list(vars, paste0("F", 1:k))
)
L_ci       <- apply(lambda_boot, 2, quantile, c(.025, .975))
psi_median <- apply(psi_boot,    2, median)

vars <- colnames(df_num)
rownames(L_median) <- vars
colnames(L_median) <- paste0("F", 1:k)
names(psi_median)  <- vars

Fnames <- paste0("F", 1:k)

df_L_ci <- do.call(
  rbind,
  lapply(seq_len(k), function(j) {
    cols <- ((j - 1) * p + 1):(j * p)  # the p columns for factor j
    data.frame(
      variable = vars,
      factor   = Fnames[j],
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
  .export   = c("align_to_ref","Lambda0","k","keep_final")
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

# c) Map each variable to its primary factor (j_star from Step 12)
j_map <- setNames(j_star, vars)                   # index 1..k for all vars
j_keep <- j_map[keep_final]
pf_keep <- paste0("F", j_keep)                   # "F1"... labels

# d) Pull the primary-factor CI row for each kept variable
primary_rows <- do.call(
  rbind,
  lapply(seq_along(keep_final), function(i) {
    v  <- keep_final[i]
    jf <- pf_keep[i]                              # factor name
    # median loading (signed) on primary from L_median_pruned
    med <- L_median_pruned[v, jf]
    # CI for primary factor from df_L_ci_pruned
    ci_row <- df_L_ci_pruned[df_L_ci_pruned$variable == v & df_L_ci_pruned$factor == jf, ]
    # safety if CI was missing (shouldn't happen if factors match)
    if (nrow(ci_row) == 0) ci_row <- data.frame(lower = NA_real_, upper = NA_real_)
    data.frame(variable = v, primary_factor = jf,
               median_loading = med,
               lower = ci_row$lower, upper = ci_row$upper,
               stringsAsFactors = FALSE)
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
# 20. Bridge EFA -> CFA (helpers + clean handoff)
# Paste this whole block right after Step 19.
# Requires: Lambda0 (pruned loadings), df_mix2_clean (full typed data)
# =============================================================================

# --- Light deps guard (don’t error if optional pkgs missing) ---
need_pkg <- function(p) if (!requireNamespace(p, quietly = TRUE))
  stop(sprintf("Package '%s' is required.", p))
invisible(lapply(c("lavaan","ggplot2","reshape2"), need_pkg))
if (!requireNamespace("semPlot", quietly = TRUE)) message("[Note] 'semPlot' not installed; path diagram skipped.")
if (!requireNamespace("foreach", quietly = TRUE)) message("[Note] 'foreach' not installed; parallel bootstraps skipped.")
if (!requireNamespace("doSNOW", quietly = TRUE)) message("[Note] 'doSNOW' not installed; parallel bootstraps skipped.")
if (!requireNamespace("parallel", quietly = TRUE)) message("[Note] 'parallel' not installed; parallel bootstraps skipped.")
if (!requireNamespace("ltm", quietly = TRUE)) message("[Note] 'ltm' not installed; point-biserial correlation skipped.")

# --- Short utilities (wrap repeated code paths) ---
safe_cfa <- function(model, data, ordered = character(0), std.lv = TRUE, est = "WLSMV") {
  tryCatch(lavaan::cfa(model, data = data, std.lv = std.lv, estimator = est, ordered = ordered),
           error = function(e) { cat("[CFA ERROR]\n"); print(e); NULL })
}
fit_print <- function(fit, label="[FIT]") {
  cat("\n", label, " Fit indices:\n", sep="")
  print(lavaan::fitMeasures(fit, c("chisq","df","cfi","rmsea","srmr")))
}
std_lambda <- function(fit) lavaan::inspect(fit, "std")$lambda
plot_resid_heatmap <- function(fit, main) {
  rr <- lavaan::residuals(fit, type = "cor")
  if (is.list(rr) && "cov" %in% names(rr)) stats::heatmap(rr$cov, main = main, symm = TRUE)
}
semplot_pdf <- function(fit, file) {
  if (!requireNamespace("semPlot", quietly = TRUE)) return(invisible(NULL))
  pdf(file, width=10, height=8); on.exit(dev.off(), add=TRUE)
  semPlot::semPaths(fit, "std", whatLabels="std", edge.label.cex=1.05, layout="tree", style="lisrel")
}
mi_refine <- function(model_txt, data, ordered, max_steps=5, th=10) {
  added <- character(0)
  fit   <- safe_cfa(model_txt, data, ordered)
  if (is.null(fit)) return(list(model=model_txt, fit=NULL, added=added))
  for (s in seq_len(max_steps)) {
    mi <- lavaan::modindices(fit)
    mi <- mi[order(-mi$mi), ]
    mi <- subset(mi, mi > th & !(op %in% c("~1","~")))
    mi <- mi[!paste(mi$lhs, mi$op, mi$rhs) %in% added, ]
    if (!nrow(mi)) { cat("\n[MI-REFINEMENT] No MI >", th, " left.\n"); break }
    top <- mi[1, ]; new_line <- paste(top$lhs, top$op, top$rhs)
    cat(sprintf("[MI-REFINEMENT] Step %d: add '%s' (MI=%.2f)\n", s, new_line, top$mi))
    model_txt <- paste(model_txt, new_line, sep="\n"); added <- c(added, new_line)
    fit2 <- safe_cfa(model_txt, data, ordered); if (!is.null(fit2)) fit <- fit2
    fit_print(fit, sprintf("[MI-REFINEMENT %d]", s))
  }
  list(model=model_txt, fit=fit, added=added)
}
collapse_sparse_categories <- function(data, var, min_count=5) {
  if (!var %in% names(data)) return(data)
  x <- data[[var]]; if (!is.factor(x) && !is.ordered(x)) return(data)
  tab <- table(x, useNA="ifany"); sparse <- names(tab)[tab < min_count]
  if (!length(sparse)) return(data)
  new <- as.character(x)
  if (is.ordered(x)) {
    lv <- levels(x); num <- suppressWarnings(as.numeric(lv))
    if (all(is.finite(num)) && length(lv) >= 4) {
      new[new %in% c("1","2")] <- "1-2"; new[new %in% c("4","5")] <- "4-5"
      data[[var]] <- factor(new, levels=c("1-2","3","4-5"), ordered=TRUE)
    } else {
      for (lev in sparse) {
        k <- which(levels(x)==lev); tgt <- ifelse(k>1, levels(x)[k-1], levels(x)[min(k+1, length(levels(x)))])
        new[new==lev] <- tgt
      }
      data[[var]] <- factor(new, levels=unique(new), ordered=TRUE)
    }
  } else {
    new[new %in% sparse] <- "Other"
    data[[var]] <- factor(new, levels=c(setdiff(levels(x), sparse), "Other"))
  }
  data
}

# --- Hand-off objects from EFA ---
cfa_vars <- rownames(Lambda0)
if (!exists("df_num")) stop("Missing 'df_num' (typed mixed-mode data) for CFA.")
missing_cfa <- setdiff(cfa_vars, names(df_num))
if (length(missing_cfa)) stop("[CFA] Variables missing in data: ", paste(missing_cfa, collapse=", "))
cfa_df <- df_num[, cfa_vars, drop=FALSE]

# Explicit lists (kept) → intersect with data
all_cont_vars <- c(
  "Q62__How_much_VEGETABLES_do_you_harvest_per_year_from_this_plot_kilograms__continuous",
  "Q50__How_much_land_that_is_yours_do_you_cultivate_bigha__continuous",
  "Q52__On_how_much_land_do_you_grow_vegetables_bigha__continuous",
  "Q109__What_is_your_households_yearly_income_overall_including_agriculture_NPR__continuous",
  "Q0__hope_total__continuous","Q0__self_control_score__continuous",
  "Q5__AgeYears__continuous","Q108__What_is_your_households_yearly_income_from_agriculture_NPR__continuous"
)
all_ordered_items <- c(
  "Q112__Generally_speaking_how_would_you_define_your_farming__ordinal",
  "Q0__average_of_farming_practices__ordinal",
  "Q70__in_the_past_12_months_did_you_receive_any_info_from_anyone_on_agriculture__binary__1"
)
cont_vars    <- intersect(all_cont_vars, names(cfa_df))
ordered_items <- intersect(all_ordered_items, names(cfa_df))

# Coerce/scale as you did
for (v in cont_vars) if (!is.numeric(cfa_df[[v]])) cfa_df[[v]] <- as.numeric(cfa_df[[v]])
if (length(cont_vars)) cfa_df[cont_vars] <- scale(cfa_df[cont_vars])
for (v in ordered_items) if (!is.ordered(cfa_df[[v]]))
  cfa_df[[v]] <- ordered(cfa_df[[v]], levels = sort(unique(cfa_df[[v]])))

# Create model syntax directly from Lambda0
cfa_model_lines <- sapply(seq_len(ncol(Lambda0)), function(j) {
  f <- colnames(Lambda0)[j]
  its <- rownames(Lambda0)[abs(Lambda0[, j]) > 0]
  if (length(its)) paste(f, "=~", paste(its, collapse = " + ")) else NULL
})
if (ncol(Lambda0) > 1) {
  pairs <- utils::combn(colnames(Lambda0), 2)
  cfa_model_lines <- c(cfa_model_lines, apply(pairs, 2, function(x) paste(x[1], "~~", x[2])))
}
cfa_model <- paste(cfa_model_lines, collapse = "\n")
cat("\n[CFA] Model syntax from EFA (Λ0):\n", cfa_model, "\n")

# =============================================================================
# 21. Robust CFA (aligned to EFA) — fit, MI refinement, diagnostics, outputs
# =============================================================================

fit_cfa <- safe_cfa(cfa_model, cfa_df, ordered_items)
if (is.null(fit_cfa)) stop("CFA failed to fit.")

fit_print(fit_cfa, "[CFA]")
cat("\n[CFA] Standardized loadings:\n"); print(std_lambda(fit_cfa))
cat("\n[CFA] Modification indices (MI > 10):\n")
print(subset(lavaan::modindices(fit_cfa), mi > 10)[, c("lhs","op","rhs","mi")])

# Save + diagram + factor scores + residual heatmap
saveRDS(fit_cfa, "fit_cfa.rds")
if (requireNamespace("semPlot", quietly = TRUE)) semplot_pdf(fit_cfa, "cfa_model_diagram.pdf")
fs <- tryCatch(lavaan::lavPredict(fit_cfa, method="Bartlett"), error=function(e) NULL)
if (!is.null(fs)) utils::write.csv(fs, "factor_scores.csv", row.names = TRUE)
plot_resid_heatmap(fit_cfa, "CFA Residual Correlation Matrix")

# MI refinement loop (unchanged logic)
mi_out <- mi_refine(cfa_model, cfa_df, ordered_items, max_steps = 5, th = 10)
cfa_model_refined <- mi_out$model; fit_cfa_refined <- mi_out$fit

cat("\n[FINAL MODEL] Standardized solution:\n")
print(summary(fit_cfa_refined, standardized = TRUE))

# Heywood cases
cat("\n[FINAL MODEL] Negative residual variances (Heywood cases):\n")
pt_ref <- lavaan::parTable(fit_cfa_refined)
neg_res <- subset(pt_ref, op=="~~" & lhs==rhs & est < 0)
if (nrow(neg_res)) print(neg_res[, c("lhs","est")]) else cat("None detected.\n")

cat("\n[FINAL MODEL] MI > 10:\n")
mif <- lavaan::modindices(fit_cfa_refined); mif <- mif[order(-mif$mi), ]
mf <- subset(mif, mi > 10 & !(op %in% c("~1","~")))
if (nrow(mf)) print(mf[, c("lhs","op","rhs","mi")]) else cat("No MI > 10 remain.\n")
plot_resid_heatmap(fit_cfa_refined, "Final CFA Residual Correlation Matrix")

# =============================================================================
# 22. Advanced diagnostics & model probes (wrapped but same operations)
# =============================================================================

# -- 1. Resolve Q70 Heywood (fix variance + distribution check) --
Q70v <- "Q70__in_the_past_12_months_did_you_receive_any_info_from_anyone_on_agriculture__binary__1"
q70_neg <- subset(pt_ref, op=="~~" & lhs==Q70v & rhs==Q70v & est < 0)
if (nrow(q70_neg)) {
  cat("\n[DIAGNOSTIC 1] Q70 Heywood detected. Fixing variance to 0.01 and refitting...\n")
  cfa_model_fixed <- paste0(cfa_model_refined, "\n", Q70v, " ~~ 0.01*", Q70v)
  fit_cfa_fixed   <- safe_cfa(cfa_model_fixed, cfa_df, ordered_items)
  if (!is.null(fit_cfa_fixed)) {
    fit_print(fit_cfa_fixed, "[Q70 FIXED]")
    ol <- std_lambda(fit_cfa_refined)[Q70v, ]; fl <- std_lambda(fit_cfa_fixed)[Q70v, ]
    cat("\nQ70 standardised loading (orig vs fixed):\n"); print(round(rbind(Original=ol, Fixed=fl), 3))
  }
  # Q70 distribution
  cat("\n[Q70] Frequency table:\n"); print(table(cfa_df[[Q70v]], useNA = "ifany"))
  cat("Proportions:\n"); print(round(prop.table(table(cfa_df[[Q70v]], useNA = "no")), 3))
} else {
  cat("\n[DIAGNOSTIC 1] Q70 Heywood not detected.\n")
}

# -- 2a. Three-factor probe --
cat("\n[DIAGNOSTIC 2a] Three-factor probe (Market-engagement / Information)...\n")
Q62 <- "Q62__How_much_VEGETABLES_do_you_harvest_per_year_from_this_plot_kilograms__continuous"
Q108<- "Q108__What_is_your_households_yearly_income_from_agriculture_NPR__continuous"
Q112<- "Q112__Generally_speaking_how_would_you_define_your_farming__ordinal"
Q109<- "Q109__What_is_your_households_yearly_income_overall_including_agriculture_NPR__continuous"
Q5  <- "Q5__AgeYears__continuous"
Q0h <- "Q0__hope_total__continuous"
Q0s <- "Q0__self_control_score__continuous"
Q52 <- "Q52__On_how_much_land_do_you_grow_vegetables_bigha__continuous"

m3 <- paste(
  paste("F1 =~", paste(c(Q62,Q108,Q0h), collapse=" + ")),
  paste("F2 =~", paste(c(Q109,Q5,Q0s), collapse=" + ")),
  paste("F3 =~", paste(c(Q112,Q70v,Q52), collapse=" + ")),
  "F1 ~~ F2\nF1 ~~ F3\nF2 ~~ F3",
  if (grepl("Q112.*Q70|Q70.*Q112", cfa_model_refined)) paste(Q112,"~~",Q70v) else "",
  sep = "\n"
)
cat(m3, "\n")
fit_cfa_3f <- safe_cfa(m3, cfa_df, ordered_items)
if (!is.null(fit_cfa_3f)) {
  fit_print(fit_cfa_3f, "[3-FACTOR]"); cat("\n[3-FACTOR] Std. loadings:\n"); print(std_lambda(fit_cfa_3f))
  c2 <- lavaan::fitMeasures(fit_cfa_refined, c("chisq","df","cfi","rmsea","srmr"))
  c3 <- lavaan::fitMeasures(fit_cfa_3f,     c("chisq","df","cfi","rmsea","srmr"))
  print(data.frame(Model=c("2-Factor","3-Factor"),
                   ChiSq=c(c2["chisq"],c3["chisq"]), DF=c(c2["df"],c3["df"]),
                   CFI=c(c2["cfi"],c3["cfi"]), RMSEA=c(c2["rmsea"],c3["rmsea"]), SRMR=c(c2["srmr"],c3["srmr"])))
}

# -- 2b. Bifactor probe --
cat("\n[DIAGNOSTIC 2b] Bifactor probe (general + specific info)...\n")
mB <- paste(
  paste("F1 =~", paste(c(Q62,Q108,Q0h,Q109,Q5,Q0s,Q112,Q70v,Q52), collapse=" + ")),
  paste("S1 =~", paste(c(Q112,Q70v,Q52), collapse=" + ")),
  "F1 ~~ 0*S1", if (grepl("Q112.*Q70|Q70.*Q112", cfa_model_refined)) paste(Q112,"~~",Q70v) else "",
  sep = "\n"
)
cat(mB, "\n")
fit_cfa_bifactor <- safe_cfa(mB, cfa_df, ordered_items)
if (!is.null(fit_cfa_bifactor)) {
  fit_print(fit_cfa_bifactor, "[BIFACTOR]"); cat("\n[BIFACTOR] Std. loadings:\n"); print(std_lambda(fit_cfa_bifactor))
}

# -- 3. Split-half cross-validation (same logic) --
set.seed(2025)
n <- nrow(cfa_df); idx <- sample(n); n2 <- floor(n/2)
s1 <- cfa_df[idx[1:n2], , drop=FALSE]; s2 <- cfa_df[idx[(n2+1):n], , drop=FALSE]
fit_s1 <- safe_cfa(cfa_model_refined, s1, ordered_items)
fit_s2 <- safe_cfa(cfa_model_refined, s2, ordered_items)
if (!is.null(fit_s1) && !is.null(fit_s2)) {
  L1 <- std_lambda(fit_s1); L2 <- std_lambda(fit_s2)
  key <- intersect(rownames(L1), rownames(L2))
  if (length(key)) {
    dif <- abs(L1[key, , drop=FALSE] - L2[key, , drop=FALSE])
    cat("\n[SPLIT-HALF] Mean |Δ loading|:", round(mean(dif),3),
        " Max |Δ|:", round(max(dif),3), "\n")
  }
}

# -- 4. Bootstrap cross-validation on refined model (kept) --
if (requireNamespace("foreach", quietly = TRUE) &&
    requireNamespace("doSNOW", quietly = TRUE) &&
    requireNamespace("parallel", quietly = TRUE)) {
  B_boot <- 100
  n_cores <- max(1, parallel::detectCores()-1)
  cl <- parallel::makeCluster(n_cores); doSNOW::registerDoSNOW(cl)
  pb <- txtProgressBar(max=B_boot, style=3)
  opts <- list(progress = function(i) setTxtProgressBar(pb, i))
  key_vars <- rownames(std_lambda(fit_cfa_refined))
  boot_loadings <- foreach::foreach(b=1:B_boot, .combine=rbind,
                                    .packages="lavaan", .options.snow=opts) %dopar% {
                                      ii <- sample(nrow(cfa_df), replace=TRUE); dat <- cfa_df[ii, ]
                                      fb <- tryCatch(lavaan::cfa(cfa_model_refined, data=dat, std.lv=TRUE, estimator="WLSMV",
                                                                 ordered=ordered_items), error=function(e) NULL)
                                      if (is.null(fb)) return(rep(NA, length(key_vars)*ncol(Lambda0)))
                                      Lb <- std_lambda(fb); as.vector(Lb[key_vars, , drop=FALSE])
                                    }
  close(pb); parallel::stopCluster(cl)
  if (ncol(as.matrix(boot_loadings)) > 0) {
    ci <- apply(boot_loadings, 2, stats::quantile, c(.025,.975), na.rm=TRUE)
    cat("\n[BOOT] Example CI range for first few loadings:\n"); print(round(ci[, 1:min(6, ncol(ci))], 3))
  }
}

# -- 5. Parameter uncertainty (+ ordered category checks) --
pt_full <- lavaan::parTable(fit_cfa_refined); pt_full$se_ratio <- abs(pt_full$est / pt_full$se)
ldg <- subset(pt_full, op=="=~"); thr <- subset(pt_full, op=="|")
large_se_loadings <- subset(ldg, se_ratio < 0.5)
large_se_thresh   <- subset(thr, se_ratio < 0.5)
cat("\n[SE] Large-SE loadings (SE ratio < .5):\n")
if (nrow(large_se_loadings)) print(large_se_loadings[, c("lhs","rhs","est","se","se_ratio")]) else cat("None\n")
cat("\n[SE] Large-SE thresholds (SE ratio < .5):\n")
if (nrow(large_se_thresh)) print(large_se_thresh[, c("lhs","rhs","est","se","se_ratio")]) else cat("None\n")
cat("\n[Ordered vars distribution]\n")
for (v in ordered_items) if (v %in% names(cfa_df)) {
  tab <- table(cfa_df[[v]], useNA="ifany"); cat("\n", v, ":\n"); print(tab)
  sp <- tab[tab < 5]; if (length(sp)) { cat("Sparse (<5):\n"); print(sp) }
}

# -- 6. Quick summary & recommendations (kept) --
cat("\n[SUMMARY] Diagnostics snapshot\n")
cat(" • Q70 Heywood:", ifelse(nrow(q70_neg)>0, "DETECTED","NOT detected"), "\n")
cat(" • 3-factor fit: ", ifelse(!is.null(fit_cfa_3f), "SUCCESS","FAILED"), "\n", sep="")
cat(" • Bifactor fit: ", ifelse(!is.null(fit_cfa_bifactor), "SUCCESS","FAILED"), "\n", sep="")

# =============================================================================
# 23. Final model refinements and lock-in (composite + collapsing + bootstraps)
# =============================================================================

# Choose final base (fixed if available)
if (exists("fit_cfa_fixed") && !is.null(fit_cfa_fixed)) {
  fit_cfa_final <- fit_cfa_fixed;  cfa_model_final <- cfa_model_fixed
  cat("\n[FINAL] Using fixed-error model for core.\n")
} else {
  fit_cfa_final <- fit_cfa_refined; cfa_model_final <- cfa_model_refined
  cat("\n[FINAL] Using MI-refined model for core.\n")
}
fit_print(fit_cfa_final, "[FINAL CORE]")

# 2a. Q70 + Q108 composite test
Q108v <- "Q108__What_is_your_households_yearly_income_from_agriculture_NPR__continuous"
if (all(c(Q70v, Q108v) %in% names(cfa_df))) {
  cfa_df$Q70_Q108_composite <- (scale(as.numeric(cfa_df[[Q70v]])) + scale(cfa_df[[Q108v]])) / 2
  # append composite to F1 line
  ml <- strsplit(cfa_model_final, "\n")[[1]]
  f1i <- grep("^F1 =~", ml); cfa_model_composite <- cfa_model_final
  if (length(f1i)) {
    ml[f1i[1]] <- paste0(ml[f1i[1]], " + Q70_Q108_composite")
    cfa_model_composite <- paste(ml, collapse="\n")
  }
  fit_cfa_composite <- safe_cfa(cfa_model_composite, cfa_df, ordered_items)
  if (!is.null(fit_cfa_composite)) {
    fit_print(fit_cfa_composite, "[COMPOSITE]")
    cat("\n[COMPOSITE] Std. loadings:\n"); print(std_lambda(fit_cfa_composite))
    f0 <- lavaan::fitMeasures(fit_cfa_final,     c("chisq","df","cfi","rmsea","srmr"))
    f1 <- lavaan::fitMeasures(fit_cfa_composite, c("chisq","df","cfi","rmsea","srmr"))
    print(data.frame(Model=c("Original","Composite"),
                     ChiSq=c(f0["chisq"],f1["chisq"]), DF=c(f0["df"],f1["df"]),
                     CFI=c(f0["cfi"],f1["cfi"]), RMSEA=c(f0["rmsea"],f1["rmsea"]), SRMR=c(f0["srmr"],f1["srmr"])))
  }
}

# 2b. Collapse sparse categories for ordered vars
cfa_df_collapsed <- cfa_df
for (v in ordered_items) cfa_df_collapsed <- collapse_sparse_categories(cfa_df_collapsed, v, min_count = 5)
ordered_items_collapsed <- names(Filter(is.ordered, cfa_df_collapsed))
fit_cfa_collapsed <- safe_cfa(if (exists("cfa_model_composite")) cfa_model_composite else cfa_model_final,
                              cfa_df_collapsed, ordered_items_collapsed)
if (!is.null(fit_cfa_collapsed)) {
  fit_print(fit_cfa_collapsed, "[COLLAPSED]")
  if (exists("fit_cfa_composite") && !is.null(fit_cfa_composite)) {
    fc <- lavaan::fitMeasures(fit_cfa_collapsed, c("chisq","df","cfi","rmsea","srmr"))
    fm <- lavaan::fitMeasures(fit_cfa_composite, c("chisq","df","cfi","rmsea","srmr"))
    print(data.frame(Model=c("Composite","Collapsed"),
                     ChiSq=c(fm["chisq"],fc["chisq"]), DF=c(fm["df"],fc["df"]),
                     CFI=c(fm["cfi"],fc["cfi"]), RMSEA=c(fm["rmsea"],fc["rmsea"]), SRMR=c(fm["srmr"],fc["srmr"])))
  }
}

# 2c. Final bootstrap (CI) on best available spec
boot_model   <- if (exists("cfa_model_composite")) cfa_model_composite else cfa_model_final
boot_data    <- if (!is.null(fit_cfa_collapsed)) cfa_df_collapsed else cfa_df
boot_ordered <- if (!is.null(fit_cfa_collapsed)) ordered_items_collapsed else ordered_items
if (requireNamespace("foreach", quietly = TRUE) &&
    requireNamespace("doSNOW", quietly = TRUE) &&
    requireNamespace("parallel", quietly = TRUE)) {
  Bf <- 200; n_cores <- max(1, parallel::detectCores()-1)
  cl <- parallel::makeCluster(n_cores); doSNOW::registerDoSNOW(cl)
  pb <- txtProgressBar(max=Bf, style=3)
  opts <- list(progress=function(i) setTxtProgressBar(pb, i))
  lam_target <- rownames(std_lambda(fit_cfa_final)); kfac <- ncol(Lambda0)
  bootL <- foreach::foreach(b=1:Bf, .combine=rbind, .packages="lavaan", .options.snow=opts) %dopar% {
    ii <- sample(nrow(boot_data), replace=TRUE); dat <- boot_data[ii, ]
    fb <- tryCatch(lavaan::cfa(boot_model, data=dat, std.lv=TRUE, estimator="WLSMV", ordered=boot_ordered), error=function(e) NULL)
    if (is.null(fb)) return(rep(NA, length(lam_target)*kfac))
    as.vector(std_lambda(fb)[lam_target, , drop=FALSE])
  }
  close(pb); parallel::stopCluster(cl)
  conv <- mean(!is.na(bootL[,1])); cat("\n[BOOT FINAL] Convergence rate:", round(conv*100,1), "%\n")
  if (conv >= .95) {
    ci <- apply(bootL, 2, stats::quantile, c(.025,.975), na.rm=TRUE)
    cat("[BOOT FINAL] 95% CIs computed.\n")
  } else cat("[BOOT FINAL] <95% convergence; interpret cautiously.\n")
}

# Pick best final model by (CFI high & RMSEA low)
best_model <- "original"; best_fit <- lavaan::fitMeasures(fit_cfa_final, c("cfi","rmsea","srmr"))
if (exists("fit_cfa_composite") && !is.null(fit_cfa_composite)) {
  fm <- lavaan::fitMeasures(fit_cfa_composite, c("cfi","rmsea","srmr"))
  if (fm["cfi"] > best_fit["cfi"] && fm["rmsea"] < best_fit["rmsea"]) { best_model <- "composite"; best_fit <- fm }
}
if (!is.null(fit_cfa_collapsed)) {
  fc <- lavaan::fitMeasures(fit_cfa_collapsed, c("cfi","rmsea","srmr"))
  if (fc["cfi"] > best_fit["cfi"] && fc["rmsea"] < best_fit["rmsea"]) { best_model <- "collapsed"; best_fit <- fc }
}
cat("\n[FINAL PICK] Best model:", best_model, " | CFI=", round(best_fit["cfi"],3),
    " RMSEA=", round(best_fit["rmsea"],3), " SRMR=", round(best_fit["srmr"],3), "\n")

saveRDS(list(
  fit_cfa_refined = fit_cfa_refined,
  fit_cfa_3f = if (exists("fit_cfa_3f")) fit_cfa_3f else NULL,
  fit_cfa_bifactor = if (exists("fit_cfa_bifactor")) fit_cfa_bifactor else NULL,
  fit_cfa_composite = if (exists("fit_cfa_composite")) fit_cfa_composite else NULL,
  fit_cfa_collapsed = if (!is.null(fit_cfa_collapsed)) fit_cfa_collapsed else NULL
), "cfa_diagnostics.rds")

# =============================================================================
# 24. Additional diagnostic checks (simple structure, residuals, collinearity)
# =============================================================================

# Issue 1: enforce simple structure for Q70 if cross-loading > .30 on F2
Lfin <- std_lambda(fit_cfa_final)
if (Q70v %in% rownames(Lfin) && "F2" %in% colnames(Lfin) && abs(Lfin[Q70v,"F2"]) > .30) {
  cat("\n[ISSUE 1] Q70 cross-loading on F2 > .30 — constraining to F1 only.\n")
  lines <- strsplit(cfa_model_final, "\n")[[1]]
  f2i <- grep("^F2 =~", lines); if (length(f2i)) lines[f2i[1]] <- gsub(paste0("\\s*\\+?\\s*", Q70v), "", lines[f2i[1]])
  f1i <- grep("^F1 =~", lines); if (length(f1i) && !grepl(Q70v, lines[f1i[1]], fixed=TRUE))
    lines[f1i[1]] <- paste0(lines[f1i[1]], " + ", Q70v)
  cfa_model_simple <- paste(lines, collapse="\n")
  fit_cfa_simple <- safe_cfa(cfa_model_simple, cfa_df, ordered_items)
  if (!is.null(fit_cfa_simple)) {
    fit_print(fit_cfa_simple, "[SIMPLE]"); sl <- std_lambda(fit_cfa_simple); cat("\n[SIMPLE] Q70 loadings:\n"); print(round(sl[Q70v, ],3))
    # small bootstrap for Q70->F1
    if (requireNamespace("foreach", quietly = TRUE) &&
        requireNamespace("doSNOW", quietly = TRUE) &&
        requireNamespace("parallel", quietly = TRUE)) {
      Bq <- 100; n_cores <- max(1, parallel::detectCores()-1)
      cl <- parallel::makeCluster(n_cores); doSNOW::registerDoSNOW(cl)
      pb <- txtProgressBar(max=Bq, style=3); opts <- list(progress=function(i) setTxtProgressBar(pb,i))
      q70b <- foreach::foreach(b=1:Bq, .combine=c, .packages="lavaan", .options.snow=opts) %dopar% {
        ii <- sample(nrow(cfa_df), replace=TRUE); dat <- cfa_df[ii, ]
        fb <- tryCatch(lavaan::cfa(cfa_model_simple, data=dat, std.lv=TRUE, estimator="WLSMV", ordered=ordered_items), error=function(e) NULL)
        if (is.null(fb)) return(NA_real_); std_lambda(fb)[Q70v,"F1"]
      }
      close(pb); parallel::stopCluster(cl)
      cat("\n[SIMPLE] Q70→F1 bootstrap 95% CI:",
          paste(round(stats::quantile(q70b, c(.025,.975), na.rm=TRUE),3), collapse=" to "), "\n")
    }
  }
} else cat("\n[ISSUE 1] Q70 simple structure OK.\n")

# Issue 2: correlated residual Q112<->Q70 necessity
pq <- lavaan::parTable(fit_cfa_final)
q112q70 <- subset(pq, op=="~~" & ((lhs==Q112 & rhs==Q70v) | (lhs==Q70v & rhs==Q112)))
if (nrow(q112q70)) {
  cat("\n[ISSUE 2] Q112↔Q70 correlated residual present.\n"); print(q112q70[, intersect(c("lhs","op","rhs","est","se","pvalue"), names(q112q70)), drop=FALSE])
  # try model without it
  ln <- strsplit(cfa_model_final, "\n")[[1]]; ln2 <- ln[!grepl("Q112.*Q70|Q70.*Q112", ln)]
  fit_noCorr <- safe_cfa(paste(ln2, collapse="\n"), cfa_df, ordered_items)
  if (!is.null(fit_noCorr)) {
    withCorr <- lavaan::fitMeasures(fit_cfa_final, c("chisq","df","cfi","rmsea","srmr"))
    noCorr   <- lavaan::fitMeasures(fit_noCorr,   c("chisq","df","cfi","rmsea","srmr"))
    print(data.frame(Model=c("WithCorr","NoCorr"),
                     ChiSq=c(withCorr["chisq"],noCorr["chisq"]), DF=c(withCorr["df"],noCorr["df"]),
                     CFI=c(withCorr["cfi"],noCorr["cfi"]), RMSEA=c(withCorr["rmsea"],noCorr["rmsea"]), SRMR=c(withCorr["srmr"],noCorr["srmr"])))
    dX <- noCorr["chisq"]-withCorr["chisq"]; dd <- noCorr["df"]-withCorr["df"]; p <- 1-stats::pchisq(dX, dd)
    cat(sprintf("Δχ²=%.3f, Δdf=%d, p=%.3f\n", dX, dd, p))
  }
} else cat("\n[ISSUE 2] No Q112↔Q70 correlated residual in final model.\n")

# Issue 3: Q70–Q108 collinearity
if (all(c(Q70v, Q108v) %in% names(cfa_df))) {
  r <- stats::cor(as.numeric(cfa_df[[Q70v]]), cfa_df[[Q108v]], use="complete.obs")
  cat("\n[ISSUE 3] Corr(Q70, Q108) =", round(r,3), "\n")
  q70n <- as.numeric(cfa_df[[Q70v]]); q108z <- scale(cfa_df[[Q108v]])[,1]
  r2 <- summary(stats::lm(q70n ~ q108z))$r.squared; vif <- 1/(1-r2)
  cat("Pseudo-VIF(Q70 ~ Q108) =", round(vif,3), "\n")
}

# =============================================================================
# 25. Final “measurement core” report (no composite; ordered/delta)
# =============================================================================

safestr <- function(x) ifelse(is.null(x)||!length(x), "NULL", paste(x, collapse=", "))
use_collapsed <- exists("cfa_df_collapsed") && !is.null(cfa_df_collapsed)
data_core     <- if (use_collapsed) cfa_df_collapsed else cfa_df
ordered_core  <- if (use_collapsed && exists("ordered_items_collapsed")) ordered_items_collapsed else ordered_items
ordered_core  <- unique(ordered_core[ordered_core %in% names(data_core)])

Q50 <- "Q50__How_much_land_that_is_yours_do_you_cultivate_bigha__continuous"
Q0a <- "Q0__average_of_farming_practices__ordinal"

cfa_model_core <- paste(
  sprintf("F1 =~ %s + %s + %s + %s + %s + %s + %s", Q50, Q52, Q62, Q108, Q112, Q0a, Q70v),
  sprintf("F2 =~ %s + %s + %s + %s + %s + %s + %s", Q5, Q52, Q109, Q0h, Q0s, Q0a, Q70v),
  paste(Q112, "~~", Q70v),
  paste(Q62,  "~~", Q0h),
  "F1 ~~ F2",
  sep = "\n"
)
cat("\n[FINAL MEASUREMENT CORE] Syntax:\n", cfa_model_core, "\n")

fit_core <- safe_cfa(cfa_model_core, data_core, ordered_core)
if (is.null(fit_core)) stop("Final core model failed.")
ptc <- lavaan::parTable(fit_core)
q70row <- subset(ptc, op=="~~" & lhs==Q70v & rhs==Q70v)
if (nrow(q70row) && isTRUE(q70row$free[1] == 1))
  stop("Q70 residual variance unexpectedly free. Ensure Q70 is ordered and WLSMV(delta).")
cat("✓ Q70 residual variance fixed by identification (ordered, delta).\n")

cat("\n[CORE] Using data: ", if (use_collapsed) "collapsed categories" else "original", "\n", sep="")
cat("[CORE] Ordered variables: ", safestr(ordered_core), "\n", sep="")
fm <- lavaan::fitMeasures(fit_core, c("chisq","df","pvalue","cfi","tli","rmsea","rmsea.ci.lower","rmsea.ci.upper","srmr"))
cat(sprintf("\n[CORE FIT] χ²(%d)=%.2f, p=%.3f | CFI=%.3f, TLI=%.3f | RMSEA=%.3f [%.3f, %.3f] | SRMR=%.3f\n",
            as.integer(fm["df"]), fm["chisq"], fm["pvalue"], fm["cfi"], fm["tli"],
            fm["rmsea"], fm["rmsea.ci.lower"], fm["rmsea.ci.upper"], fm["srmr"]))

Lam <- std_lambda(fit_core)
cat("\n[CORE LOADINGS] (|λ|>.30)\n")
if ("F1" %in% colnames(Lam)) print(round(Lam[abs(Lam[,"F1"])>.30,"F1", drop=FALSE],3))
if ("F2" %in% colnames(Lam)) print(round(Lam[abs(Lam[,"F2"])>.30,"F2", drop=FALSE],3))
Psi <- lavaan::inspect(fit_core, "std")$psi
if (!is.null(Psi)) cat(sprintf("\nFactor correlation (F1↔F2): %.3f\n", Psi["F1","F2"]))

# Simple reliability (composite-like)
comp_rel <- function(v) { v <- v[is.finite(v)]; v <- v[abs(v)>.30]; if (length(v)<2) return(NA_real_); sl<-sum(v); sl2<-sum(v^2); se<-length(v)-sl2; (sl^2)/((sl^2)+se) }
if ("F1" %in% colnames(Lam)) cat(sprintf("[RELIABILITY] F1: %s\n", ifelse(is.na(comp_rel(Lam[,"F1"])),"NA", round(comp_rel(Lam[,"F1"]),3))))
if ("F2" %in% colnames(Lam)) cat(sprintf("[RELIABILITY] F2: %s\n", ifelse(is.na(comp_rel(Lam[,"F2"])),"NA", round(comp_rel(Lam[,"F2"]),3))))

cat("\n[FINAL NOTE] Q112↔Q70 and Q62↔hope residual covariances retained if theory-consistent.\n")
cat("\n=== CFA pipeline complete. Ready for reporting / structural modeling. ===\n")



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

