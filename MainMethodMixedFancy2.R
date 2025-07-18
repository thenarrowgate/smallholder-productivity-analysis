# Step 1 ─ Load required packages
library(dplyr)       # data wrangling
library(EFAtools)    # VSS(), tenBerge scores
library(boot)        # bootstrap()
library(Gifi)        # princals()
library(lavaan)      # sem()
library(mgcv)        # gam()
library(polycor)     # hetcor()
library(psych)       # mixedCor(), fa.*, factor.congruence(), factor.scores
library(readxl)      # read_excel()
library(stringr)     # str_split()
library(doSNOW)      # parallel backend
library(foreach)     # foreach looping
library(ggplot2)     # plotting
library(reshape2)    # melt()

# Step 2 ─ Set seed and working directory
set.seed(2025)
LOCAL_DIR <- "E:/Atuda/67814-Data-Science-Final-Project/Code"
setwd(LOCAL_DIR)

# Step 3 ─ Read in data and drop outcome columns
df <- read_excel("nepal_dataframe_FA.xlsx")
y_prod <- df$Q0__AGR_PROD__continuous
df     <- df %>% select(-Q0__AGR_PROD__continuous,
                        -Q0__sustainable_livelihood_score__continuous)

# Step 4 ─ Split variables by type
types <- str_split(names(df), "__", simplify = TRUE)[,3]
types[types == "binary_nominal"] <- "nominal"
df_cont <- df[, types == "continuous", drop = FALSE]
df_ord  <- df[, types == "ordinal",    drop = FALSE]
df_bin  <- df[, types == "binary",     drop = FALSE]
df_nom  <- df[, types == "nominal",    drop = FALSE]

# Step 5 ─ Convert ordinal/binary to ordered factors
df_ord_factored <- df_ord %>% mutate(across(everything(), ordered))
df_bin_factored <- df_bin %>% mutate(across(everything(), ordered))

# Step 6 ─ Rebuild mixed‐type dataset and drop NAs
df_mix2       <- bind_cols(df_cont, df_ord_factored, df_bin_factored)
df_mix2_clean <- df_mix2[, colSums(is.na(df_mix2)) == 0]

# Step 7 ─ Debug: drop unsupported column classes
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


# Step 8 ─ Compute mixed correlation matrix and compare eigenvalues
het_out <- hetcor(df_mix2_clean, use = "pairwise.complete.obs")
R_mixed <- het_out$correlations
stopifnot(!any(is.na(R_mixed)))

ev_raw <- eigen(hetcor(df_mix2_clean)$correlations)$values
ev_adj <- eigen(R_mixed)$values
plot(ev_raw, ev_adj, main="Eigenvalue comparison")

# Step 9 ─ Determine number of factors (parallel analysis & MAP)
pa_out <- fa.parallel(R_mixed, n.obs=nrow(df_mix2_clean),
                      fm="minres", fa="fa",
                      n.iter=500, quant=.95,
                      cor="cor", use="pairwise", plot=FALSE)
k_PA  <- pa_out$nfact
vss_out <- VSS(R_mixed, n=ncol(R_mixed),
               fm="minres", n.obs=nrow(df_mix2_clean), plot=FALSE)
k_MAP <- which.min(vss_out$map)
k     <- k_MAP  # choose k

# Step 10 ─ Bootstrap robust MINRES+oblimin to get loadings & uniquenesses
p <- ncol(df_mix2_clean)
B <- 1000
n_cores <- parallel::detectCores() - 1
cl <- makeCluster(n_cores); registerDoSNOW(cl)
pb <- txtProgressBar(max=B, style=3)
opts <- list(progress = function(n) setTxtProgressBar(pb, n))

boot_load <- foreach(b=1:B, .combine=rbind,
                     .packages=c("psych","polycor"),
                     .options.snow=opts) %dopar% {
                       repeat {
                         samp <- df_mix2_clean[sample(nrow(df_mix2_clean), replace=TRUE), ]
                         Rb   <- tryCatch(hetcor(samp)$correlations, error=function(e) NULL)
                         if(is.null(Rb) || any(is.na(Rb))) next
                         fa_b <- tryCatch(fa(Rb, nfactors=k, fm="minres", rotate="oblimin", n.obs=nrow(samp)),
                                          error=function(e) NULL)
                         if(is.null(fa_b)) next
                         return(c(as.vector(fa_b$loadings[]), fa_b$uniquenesses))
                       }
                     }
close(pb); stopCluster(cl)

# Step 11 ─ Summarize bootstrap: medians & 95% CIs
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

# build df_psi_ci for the uniqueness‐CI
psi_ci   <- apply(psi_boot, 2, quantile, c(.025, .975))
df_psi_ci <- data.frame(
  variable = vars,
  lower    = psi_ci[1, ],
  upper    = psi_ci[2, ],
  stringsAsFactors = FALSE
)

# Write medians and CIs to CSV
write.csv(L_median,        "L_median_sngl.csv",   row.names = TRUE)
write.csv(df_L_ci,         "L_ci_long_sngl.csv",  row.names = FALSE)
write.csv(data.frame(variable = vars,
                     psi_median = psi_median),
          "psi_median_sngl.csv", row.names = FALSE)
write.csv(df_psi_ci,       "psi_ci_sngl.csv",     row.names = FALSE)

# Reload L_median (matrix, preserving row names)
L_median <- as.matrix(
  read.csv("L_median_sngl.csv",
           row.names   = 1,
           check.names = FALSE)
)

vars <- rownames(L_median)

# Reload df_L_ci
df_L_ci <- read.csv("L_ci_long_sngl.csv",
                    stringsAsFactors = FALSE,
                    check.names      = FALSE)

# Reload psi_median (named vector)
psi_tmp    <- read.csv("psi_median_sngl.csv", stringsAsFactors = FALSE,
                       check.names = FALSE)
psi_median <- psi_tmp$psi_median
names(psi_median) <- psi_tmp$variable

# Step 12 ─ Prune items via decision-tree rules
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
for(i in seq_len(nrow(Lambda0))) {
  row <- Lambda0[i,]; idx <- order(abs(row), decreasing=TRUE)
  sec <- idx[2]
  if(abs(row[sec])<.15) Lambda0[i,sec] <- 0
}

R_prune <- R_mixed[keep, keep]

# Step 13 ─ Prune survivors with low communality (h²<.25)
h2   <- rowSums(Lambda0^2)
drop_comm <- names(h2)[h2<0]
if(length(drop_comm)) message("Dropping low-h² (<.23): ", paste(drop_comm, collapse=", "))
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
n_cores <- parallel::detectCores() - 1
cl      <- makeCluster(n_cores)
registerDoSNOW(cl)

pb       <- txtProgressBar(max = B, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts     <- list(progress = progress)

# parallel bootstrap + compute φ and H
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

# tear down
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

# ## Step 20 ─ Extract final factor scores
# # 1. Convert all retained columns to numeric (ordered factors → integer codes)
raw_data <- df_mix2_clean[, keep_final, drop = FALSE]
raw_data_num <- as.data.frame(lapply(raw_data, function(col) {
  if (is.numeric(col)) {
    col
  } else if (is.factor(col) || is.ordered(col)) {
    as.numeric(col)
  } else {
    stop("Column not numeric or factor: ", deparse(substitute(col)))
  }
}))

# ============  1.  MANUAL TEN-BERGE FACTOR SCORES  ============

# 1.1  Final data matrix in analysis order (pruned vars only)
X_full  <- raw_data_num[, keep_final]               # raw

# helper: turn each column into the numeric form used by hetcor() ---
numify <- function(x) {
  if (is.numeric(x) || is.integer(x)) {
    return(x)                                # leave numeric as-is
  } else if (is.ordered(x) || is.factor(x)) {
    return(as.numeric(x))                    # 1,2,3... coding (hetcor default)
  } else {
    stop("Unexpected column class: ", class(x))
  }
}
X_num <- as.data.frame(lapply(X_full, numify))

X_std   <- scale(X_num)                  # z-scores (μ=0, σ=1)

# 1.2  Mixed-type (poly/pearson) R for pruned vars
Rzz     <- R_prune                                   # already computed

# 1.3  Ten Berge weight matrix  W  =  R⁻¹ Λ (Λᵀ R⁻¹ Λ)⁻¹
Rinv    <- solve(Rzz)
A       <- Rinv %*% Lambda0
Tmat    <- t(Lambda0) %*% Rinv %*% Lambda0
Wten    <- A %*% solve(Tmat)

# 1.4  Scores  F̂  =  X_std  ×  W
F_hat   <- X_std %*% Wten
colnames(F_hat) <- colnames(Lambda0)
rownames(F_hat) <- rownames(df_mix2_clean)           # keep farmer IDs

# ============  2.  TREAT NOMINAL PREDICTORS  ============

# Helper to collapse rare levels (<5 %)
collapse_rare <- function(f, threshold = 0.05) {
  tab <- prop.table(table(f))
  levels_to_collapse <- names(tab)[tab < threshold]
  f_new <- as.character(f)
  f_new[f_new %in% levels_to_collapse] <- "Other"
  factor(f_new)
}

# 2.1  Prepare nominal block with collapsing
nom_raw <- df_nom
nom_collapsed <- nom_raw %>%
  mutate(across(everything(), collapse_rare))

# 2.2  Tag each nominal as “large” (n_levels > 10) or “small”
is_large <- function(v) nlevels(v) > 10
large_noms  <- names(Filter(is_large,  nom_collapsed))
small_noms  <- names(Filter(Negate(is_large), nom_collapsed))

# ============  3.  BUILD AND FIT  mgcv::gam  ============

library(mgcv)

# 3.1  Assemble analysis frame for GAM
gam_df <- data.frame(
  prod_index = y_prod,
  F_hat,
  nom_collapsed
)

# 3.2  Construct formula-text dynamically
smooth_terms   <- paste0("s(", colnames(F_hat), ")")
small_terms    <- if (length(small_noms))
  paste(small_noms, collapse = " + ") else NULL
large_terms    <- if (length(large_noms))
  paste0("s(", large_noms, ", bs='re')", collapse = " + ") else NULL

all_terms <- c(smooth_terms, small_terms, large_terms)
gam_form  <- as.formula(
  paste("prod_index ~", paste(all_terms, collapse = " + "))
)

# 3.3  Fit with REML (automatic shrinkage)
gam_fit <- gam(gam_form, data = gam_df, method = "REML", select = TRUE)

# 3.4  Print EDF table and smooth significance
print(summary(gam_fit)$s.table)      # EDF, F, p for smooths
print(summary(gam_fit)$p.table)      # parametric terms

# 3.5  Spine plots of smooths
par(mfrow = c(1, k))                 # k = 2 factors here
plot(gam_fit, pages = 1, all.terms = TRUE, shade = TRUE)

# # ============  4.  BAYESIAN LP-SEM  (brms)  ============
# 
# library(brms)
# 
# ## 4·1  Decide which factors need curvature (EDF ≥ 1.5 in GAM)
# edf_info       <- summary(gam_fit)$s.table
# spline_factors <- rownames(edf_info)[edf_info[ , "edf"] >= 1.5]
# spline_factors <- gsub("^s\\(|\\)$", "", spline_factors)        # drop “s(” & “)”
# spline_factors <- make.names(spline_factors, unique = TRUE)     # safe names
# 
# ## 4·2  Build polynomial‐score data frame
# poly_df <- data.frame(matrix(nrow = nrow(F_hat), ncol = 0))
# rownames(poly_df) <- rownames(F_hat)
# for (f in spline_factors) {
#   poly_df[[paste0(f, "_sq")]] <- F_hat[ , f]^2
#   poly_df[[paste0(f, "_cu")]] <- F_hat[ , f]^3
# }
# 
# ## 4·3  Dummy‐code collapsed nominals
# dummy_df <- model.matrix(~ . - 1, nom_collapsed) |> as.data.frame()
# dummy_df <- dummy_df[sapply(dummy_df, var) > 0]
# 
# ## 4·4  Assemble data for brms
# brms_df <- data.frame(
#   prod_index = y_prod,
#   as.data.frame(F_hat),
#   poly_df,
#   dummy_df
# )
# 
# ## 4·5  Sanitize variable names (no "__" or trailing "_")
# clean_names <- function(x) {
#   x <- gsub("__+", "_", x)       # collapse double underscores
#   x <- gsub("_+$", "", x)        # drop trailing underscores
#   make.names(x, unique = TRUE)   # ensure syntactic validity
# }
# names(brms_df) <- clean_names(names(brms_df))
# 
# ## 4·6  Build the formula
# all_vars <- setdiff(names(brms_df), "prod_index")
# # identify factor score names: F1, F2, ..., Fk
# factor_names <- intersect(all_vars, paste0("F", seq_len(k)))
# # spline terms where indicated
# factor_terms <- vapply(factor_names, function(f) {
#   if (f %in% spline_factors) paste0("s(", f, ")") else f
# }, character(1))
# # polynomial score names end in _sq or _cu
# poly_names <- grep("(_sq|_cu)$", all_vars, value = TRUE)
# # the rest are dummies
# dummy_names2 <- setdiff(all_vars, c(factor_names, poly_names))
# # assemble RHS
# rhs_terms <- c(factor_terms, poly_names, dummy_names2)
# brms_form <- as.formula(paste("prod_index ~", paste(rhs_terms, collapse = " + ")))
# 
# ## 4·7  Specify priors
# priors <- c(
#   prior(normal(0,  .3), class = "b"),    # linear coefficients
#   prior(normal(0,  .2), class = "sds")   # smooth terms’ scale parameters
# )
# 
# ## 4·8  Fit the Bayesian model in brms
# fit_brms <- brm(
#   formula = brms_form,
#   data    = brms_df,
#   family  = gaussian(),
#   prior   = priors,
#   chains  = 4,
#   cores   = 4,
#   iter    = 4000,
#   control = list(adapt_delta = 0.95),
#   seed    = 2025
# )
# 
# print(summary(fit_brms))
# 
# ## 4·9  Posterior predictive check and LOO
# pp_check(fit_brms)
# loo_brms <- loo(fit_brms)
# print(loo_brms)

# ============  4.  BAYESIAN LP-SEM  (blavaan)  ============

library(blavaan)

## 4·1  Decide which factors need curvature (EDF ≥ 1.5 in GAM)
edf_info       <- summary(gam_fit)$s.table
spline_factors <- rownames(edf_info)[edf_info[ , "edf"] >= 1.5]
spline_factors <- gsub("^s\\(|\\)$", "", spline_factors)
spline_factors <- make.names(spline_factors, unique = TRUE)

## 4·2  Polynomial-score columns
poly_df <- data.frame(row.names = rownames(F_hat))
for (f in spline_factors) {
  poly_df[[paste0(f, "_sq")]] <- F_hat[ , f]^2
  poly_df[[paste0(f, "_cu")]] <- F_hat[ , f]^3
}
names(poly_df) <- make.names(names(poly_df), unique = TRUE)

## 4·3  Dummy-code collapsed nominals (no intercept)
dummy_mat  <- model.matrix(~ . , nom_collapsed)[ , -1, drop = FALSE]
dummy_df   <- dummy_mat[ , apply(dummy_mat, 2, var) > 0, drop = FALSE]
dummy_df   <- as.data.frame(dummy_df)
names(dummy_df) <- make.names(names(dummy_df), unique = TRUE)
dummy_names     <- names(dummy_df)

## 4·4  Manifest indicators  (prune collinear pairs, scale numerics)
ind_df <- df_mix2_clean[ , keep_final, drop = FALSE]

num_vars <- names(ind_df)[sapply(ind_df, is.numeric)]
R_ind    <- cor(ind_df[num_vars], use = "pairwise.complete.obs")
hc       <- which(abs(R_ind) > .99 & abs(R_ind) < 1, arr.ind = TRUE)
if (nrow(hc)) {
  to_drop   <- unique(colnames(R_ind)[hc[ , 2]])
  message(">> Dropping near-perfectly correlated indicators: ",
          paste(to_drop, collapse = ", "))
  ind_df    <- ind_df[ , setdiff(names(ind_df), to_drop), drop = FALSE]
  keep_final <- setdiff(keep_final, to_drop)
}
num_vars <- intersect(num_vars, names(ind_df))
ind_df[num_vars] <- lapply(ind_df[num_vars], scale)

## 4·5  Assemble SEM data
sem_df <- cbind(ind_df,
                prod_index = y_prod,
                poly_df,
                dummy_df)

## 4·6  -------- Rank check for the exogenous block ---------------
xmat <- model.matrix(~ . , data = cbind(poly_df, dummy_df))
cat(">> Initial X  : rank =", qr(xmat)$rank,
    " / p =", ncol(xmat), "\n")

if (qr(xmat)$rank < ncol(xmat)) {
  keep_cols  <- qr(xmat)$pivot[seq_len(qr(xmat)$rank)]
  drop_cols  <- setdiff(colnames(xmat), colnames(xmat)[keep_cols])
  cat(">> Dropping redundant exogenous columns:\n   ",
      paste(drop_cols, collapse = ", "), "\n")
  poly_df    <- poly_df[ , setdiff(names(poly_df),  drop_cols), drop = FALSE]
  dummy_df   <- dummy_df[ , setdiff(names(dummy_df), drop_cols), drop = FALSE]
  dummy_names <- names(dummy_df)
  sem_df     <- cbind(ind_df,
                      prod_index = y_prod,
                      poly_df,
                      dummy_df)
}
## -----------------------------------------------------------------

## 4·7  Measurement block  – Λ₀ & Ψ₀ are *fixed constants*
meas_lines <- lapply(seq_len(k), function(j) {
  f  <- paste0("F", j)
  nz <- which(Lambda0[ , j] != 0)
  paste0(f, " =~ ",
         paste(sprintf("%.5f*%s", Lambda0[nz, j], rownames(Lambda0)[nz]),
               collapse = " + "))
})

resid_lines <- sprintf("%s ~~ %.5f*%s",
                       rownames(Lambda0),
                       psi_median[ rownames(Lambda0) ],
                       rownames(Lambda0))

cov_line <- if (k > 1)
  sprintf("F1 ~~ %.5f*F2", phi_mean[1]) else ""

meas_block <- paste(c(meas_lines, resid_lines, cov_line), collapse = "\n")

## 4·8  Structural block
struct_terms <- c(paste0("F", 1:k), names(poly_df), dummy_names)
struct_line  <- paste("prod_index ~", paste(struct_terms, collapse = " + "))

## 4·9  Full lavaan syntax
model_syn <- paste(meas_block, "# structural", struct_line, sep = "\n")

## 4·10  Allowed priors for Stan
priors <- dpriors(lambda = "normal(0,1)")

## 4·11  Fit Bayesian LP-SEM
ordered_vars <- intersect(names(df_ord_factored), colnames(sem_df))

library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

fit_full <- bsem(
  model     = model_syn,
  data      = sem_df,
  fixed.x   = TRUE,
  std.lv    = TRUE,
  target    = "stan",
  dp        = priors,
  cp        = "srs",
  inits     = "lavaan",
  n.chains  = 4,
  burnin    = 1000,    # warm-up for Stan
  sample    = 2000,    # post-warm-up draws per chain
  bcontrol  = list(
    thin    = 1,       # thinning rate
    control = list(
      adapt_delta   = 0.90,  # Stan HMC tuning
      max_treedepth = 8
    )
  ),
  seed      = 2025,
  ordered   = ordered_vars
)


## 4·12  Diagnostics
print(summary(fit_full), digits = 3)
cat("\nPosterior-predictive p-value:", ppp(fit_full), "\n")
loo_curved <- blavInspect(fit_full, "loo")
print(loo_curved)

# pull out the summary table with pi.lower / pi.upper
post_tab <- parameterEstimates(fit_full, standardized = FALSE)

# keep only the “free” loadings & regressions whose 95% CI excludes 0
sig_pars <- subset(post_tab,
                   (!label %in% c("start","==")) &   # drop any fixed‐start entries
                     ( (ci.lower > 0 & ci.upper > 0) |
                         (ci.lower < 0 & ci.upper < 0) ) )

cat("\n--- Significant parameters (95% CI excludes zero) ---\n")
print(sig_pars[, c("lhs","op","rhs","est","ci.lower","ci.upper")])

# now *construct replicability* H for each factor
#   H_j = ( sum_i λ_{ij} )^2 / [ (sum_i λ_{ij})^2 + sum_i θ_i ]
# where θ_i = residual variance of manifest i
#
# we’ll do that on the *posterior mean* here, which is
#   a quick approximation to the “in each replicate” step.

# pull out posterior means of loadings & residual variances
lam_means <- setNames(
  post_tab$est[ post_tab$op=="=~" ], 
  paste0(post_tab$lhs[post_tab$op=="=~"], "__", post_tab$rhs[post_tab$op=="=~"])
)
theta_means <- setNames(
  post_tab$est[ post_tab$op=="~~" & post_tab$lhs==post_tab$rhs ],
  post_tab$lhs[ post_tab$op=="~~" & post_tab$lhs==post_tab$rhs ]
)

H_by_factor <- sapply(paste0("F",1:k), function(f) {
  # all λ_{i,f}
  lambdas <- lam_means[ grep(paste0("^",f,"__"), names(lam_means)) ]
  numer   <- sum(lambdas)^2
  denom   <- numer + sum(theta_means[names(lambdas)])
  numer/denom
})
cat("\nConstruct replicability H (posterior‐mean approx.):\n")
print(H_by_factor)

ppp_val <- ppp(fit_full)
cat("\nPosterior‐predictive p‐value (PPP) =", round(ppp_val,3), "\n")

# LOO (approximate out‐of‐sample fit)
loo_vals <- blavInspect(fit_full, "loo")
print(loo_vals)

# (Optionally, if you really want the classical χ² / CFI / RMSEA / SRMR 
#  computed at the posterior‐mean estimates, you can do:)
library(lavaan)
# rebuild a lavaan object at the posterior‐mean covariance & means
Sigma_hat <- blavInspect(fit_full, "cov.ov")
mu_hat    <- blavInspect(fit_full, "mean.ov")
n         <- blavInspect(fit_full, "nobs")
lav_mod   <- sem(model_syn, sample.cov = Sigma_hat, 
                 sample.mean = mu_hat, sample.nobs = n,
                 std.lv = TRUE, fixed.x = TRUE)
cat("\nClassical fit indices at posterior‐mean solution:\n")
print(fitMeasures(lav_mod, c("chisq","df","cfi","tli","rmsea","srmr")))

# ## ──────────  Quick 30-second pilot for bsem()  ──────────
# library(blavaan)            # ≥ 0.3-15
# library(rstan)              # for rstan_options()
# # enable parallel chains
# options(mc.cores = parallel::detectCores())
# rstan_options(auto_write = TRUE)
# 
# # Only the priors Stan accepts when you freeze Λ₀ & Ψ₀
# pri_ok <- dpriors(lambda = "normal(0,1)")
# 
# # Which variables are ordinal?
# ord_ok <- intersect(names(df_ord_factored), colnames(sem_df))
# 
# # Fast pilot: 4 chains, 200 warmup, 400 sample, shallow trees
# fit_quick <- bsem(
#   model     = model_syn,
#   data      = sem_df,
#   fixed.x   = TRUE,
#   std.lv    = TRUE,
#   target    = "stan",
#   dp        = pri_ok,
#   cp        = "srs",           # shrink any remaining free covariances
#   inits     = "lavaan",        # admissible WLSMV starts internally
#   n.chains  = 4,
#   burnin    = 200,             # warmup iterations
#   sample    = 400,             # post-warmup draws per chain
#   bcontrol  = list(
#     thin    = 1,               
#     control = list(
#       adapt_delta   = 0.80,    # faster adaptation target :contentReference[oaicite:7]{index=7}
#       max_treedepth =  6       # fewer leapfrog steps per proposal :contentReference[oaicite:8]{index=8}
#     )
#   ),
#   seed      = 2025,
#   ordered   = ord_ok
# )
# 
# # Quick convergence check
# rhat <- blavInspect(fit_quick, "psrf")[, "point.est"]
# if (all(rhat < 1.05, na.rm=TRUE)) {
#   message("👍 Pilot OK – all R̂ < 1.05")
# } else {
#   message("⚠️  Some R̂ ≥ 1.05; consider more iterations or adjust adapt_delta")
# }
# 

