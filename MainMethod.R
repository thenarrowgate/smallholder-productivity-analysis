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
library(WGCNA)   # provides bicor()
library(Matrix)   # nearPD for KMO/Bartlett

# Step 2 ─ Set seed and working directory
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


# ── Step 8 ─ Compute correlation matrix ---------------------------------------
# Choose between "mixed" (default) or "spearman" correlations
COR_METHOD <- "mixed"

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
  
  # Ensure positive-definite matrix for suitability tests
  R_mixed <- as.matrix(nearPD(R_mixed, corr=TRUE)$mat)
}

stopifnot(!any(is.na(R_mixed)))
# Step 8b - Suitability checks: KMO and Bartlett tests
kmo_res <- psych::KMO(R_mixed)
bart_res <- psych::cortest.bartlett(R_mixed, n = nrow(df_mix2_clean))
cat("KMO overall MSA:", round(kmo_res$MSA, 3), "\n")
cat("Bartlett's test p-value:", signif(bart_res$p.value, 3), "\n")


if (COR_METHOD == "mixed") {
  ev_raw <- eigen(hetcor(df_mix2_clean, use = "pairwise.complete.obs")$correlations)$values
} else {
  df_num_ev <- as.data.frame(lapply(df_mix2_clean, as.numeric))
  ev_raw <- eigen(cor(df_num_ev, method = "spearman", use = "pairwise.complete.obs"))$values
}
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

# Step 10 ─ Bootstrap robust MINRES+geomin to get loadings & uniquenesses
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
                         Rb   <- tryCatch({
                           if (COR_METHOD == "mixed") {
                             hetcor(samp, use = "pairwise.complete.obs")$correlations
                           } else {
                             samp_num <- as.data.frame(lapply(samp, as.numeric))
                             cor(samp_num, method = "spearman", use = "pairwise.complete.obs")
                           }
                         }, error=function(e) NULL)
                         if(is.null(Rb) || any(is.na(Rb))) next
                         fa_b <- tryCatch(fa(Rb, nfactors=k, fm="minres", rotate="geominQ", n.obs=nrow(samp)),
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

# Step 13 ─ Prune survivors with low communality (h²<.20)
h2   <- rowSums(Lambda0^2)
drop_comm <- names(h2)[h2<0.2]
if(length(drop_comm)) message("Dropping low-h² (<.2): ", paste(drop_comm, collapse=", "))
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
phi_mean <- colMeans(phis_rob)
H_mean   <- colMeans(Hs_rob)

cat(sprintf("Finished %d valid bootstraps\n", nrow(phis_rob)))
cat("Robust mean Tucker's φ: ", phi_mean, "\n")
cat("Robust mean Hancock's H:",  H_mean,   "\n")



