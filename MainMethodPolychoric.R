# load required packages
library(readxl)    # for read_excel()
library(dplyr)     # for data wrangling
library(stringr)   # for str_split()
library(psych)     # for mixedCor(), fa.parallel(), fa(), factor.congruence()
library(EFAtools)  # for VSS()
library(boot)      # for bootstrap()
library(mgcv)      # for gam()
library(lavaan)    # for sem()


set.seed(2025)

# 1. Load data
df <- read_excel("nepal_dataframe_FA.xlsx")

# Extract outcomes
y_prod <- df$Q0__AGR_PROD__continuous
df      <- df %>% select(-Q0__AGR_PROD__continuous,
                         -Q0__sustainable_livelihood_score__continuous)

# 2. Split into continuous / ordinal / binary
types   <- str_split(names(df), "__", simplify = TRUE)[,3]
types[types=="binary_nominal"] <- "nominal"
df_cont <- df[, types=="continuous", drop = FALSE]
df_ord  <- df[, types=="ordinal",    drop = FALSE]
df_bin  <- df[, types=="binary",     drop = FALSE]

# ────────────────────────────────────────────────────────────────────────────────
# 2a. Pull out all nominal variables
df_nom     <- df[, types == "nominal", drop = FALSE]

# 2b. Find those with >6 unique levels
nom_levels <- sapply(df_nom, function(x) length(unique(x)))
high_nom   <- names(nom_levels[nom_levels > 6])

if (length(high_nom) > 0) {
  # Ensure Gifi is loaded
  library(Gifi)
  
  # 1. Convert to factors
  df_nom[high_nom] <- lapply(df_nom[high_nom], factor)
  
  # 2. Run categorical PCA (princals) on all high-cardinal nominals at once
  m      <- length(high_nom)
  pc_out <- princals(
    data    = as.data.frame(df_nom[high_nom]), 
    ndim    = m, 
    levels  = rep("nominal", m),
    verbose = FALSE
  )
  
  # 3. Sanity-check that object scores exist (n × m)
  if (!"objectscores" %in% names(pc_out)) {
    stop("princals() did not return 'objectscores'!")
  }
  dims <- dim(pc_out$objectscores)
  if (!all(dims == c(nrow(df_nom), m))) {
    stop(sprintf(
      "Unexpected objectscores dimensions: got %dx%d but expected %dx%d",
      dims[1], dims[2], nrow(df_nom), m
    ))
  }
  
  # 4. Extract the m-dimensional object scores as new numeric columns
  df_quant <- as.data.frame(pc_out$objectscores)
  names(df_quant) <- paste0(high_nom, "_quant")
  
  # 5. Append to df_cont and drop originals
  df_cont <- bind_cols(df_cont, df_quant)
  df      <- df %>% select(-all_of(high_nom))
  df_nom  <- df_nom[, setdiff(names(df_nom), high_nom), drop = FALSE]
}
# ────────────────────────────────────────────────────────────────────────────────

# Identify which of the new _quant columns have near-zero variance
quant_cols <- grep("_quant$", names(df_cont), value = TRUE)
variances <- sapply(df_cont[quant_cols], var, na.rm = TRUE)

# Drop any that are essentially constant
zero_var <- names(variances)[variances < 1e-6]
if (length(zero_var) > 0) {
  warning("Dropping near-constant quant cols: ", paste(zero_var, collapse = ", "))
  df_cont <- df_cont %>% select(-all_of(zero_var))
  quant_cols <- setdiff(quant_cols, zero_var)
}

# Standardize the remaining quant columns
df_cont[quant_cols] <- lapply(df_cont[quant_cols], scale)

# install.packages("polycor")  # if not already installed
library(polycor)    # for hetcor()

# 1. Convert ordinal & binary predictors to ordered factors
df_ord_factored <- df_ord %>%
  mutate(across(everything(), ~ ordered(.)))
df_bin_factored <- df_bin %>%
  mutate(across(everything(), ~ ordered(.)))

# 2. Re‐build the mixed‐type data frame
df_mix2 <- bind_cols(df_cont, df_ord_factored, df_bin_factored)

# 3. Drop any columns with missing values
df_mix2_clean <- df_mix2[, colSums(is.na(df_mix2)) == 0]

# ────────────────────────────────────────────────────────────────────────────────
# 5. Debug + auto‐clean before hetcor
# ────────────────────────────────────────────────────────────────────────────────

# Assume df_mix2_clean already contains the bind_cols(df_cont, df_ord_factored, df_bin_factored)
# and has had NA‐only columns removed.

# 5a) Print out every column’s class
cat(">>> DEBUG: column classes:\n")
print( sapply(df_mix2_clean, class) )

# 5b) Identify which columns are NOT numeric, integer, factor, ordered, logical, or character
allowed <- function(x) {
  any(inherits(x, c("numeric","integer","factor","ordered","logical","character")))
}
good_cols <- vapply(df_mix2_clean, allowed, logical(1))
bad_cols  <- names(good_cols)[!good_cols]

if(length(bad_cols)>0) {
  cat(">>> WARNING: dropping these unsupported columns:\n")
  print(bad_cols)
  df_mix2_clean <- df_mix2_clean[ , good_cols, drop = FALSE ]
}

# 5c) (Optional) if you’d rather try coercion instead of dropping:
# for(col in bad_cols) {
#   df_mix2_clean[[col]] <- as.numeric(unlist(df_mix2_clean[[col]]))
# }

# Verify post‐clean
cat(">>> POST‐CLEAN: column classes:\n")
print( sapply(df_mix2_clean, class) )

# Convert from tibble to plain data.frame so [ , i] yields a numeric/factor vector:
df_mix2_clean <- as.data.frame(df_mix2_clean)

# (Optional) verify:
cat("Post‐conversion class:", class(df_mix2_clean), "\n")  
# Should print "data.frame"

# ────────────────────────────────────────────────────────────────────────────────
# 6. Now compute the mixed (heterogeneous) correlation matrix
# ────────────────────────────────────────────────────────────────────────────────


# 4. Compute the heterogeneous correlation matrix
het_out <- hetcor(
  df_mix2_clean,
  use = "pairwise.complete.obs"    # pairwise deletion for missing
)
R_mixed <- het_out$correlations    # the correlation matrix

any(is.na(R_mixed))  # should be FALSE

ev_raw <- eigen(hetcor(df_mix2_clean)$correlations)$values
ev_adj <- eigen(R_mixed)$values
plot(ev_raw, ev_adj, main="Eigenvalue comparison")


# 7. Parallel analysis & MAP to pick k
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

# 8. Initial MINRES + oblimin
efa_init <- fa(R_mixed, nfactors = k,
               fm = "minres", rotate = "oblimin", n.obs = n)

# 9. Prune low‐communality items
h2      <- efa_init$communality
keep    <- names(h2)[h2 >= .30]
R_prune <- R_mixed[keep, keep]
efa0    <- fa(R_prune, nfactors = k,
              fm = "minres", rotate = "oblimin", n.obs = n)
Lambda0 <- efa0$loadings
Psi0    <- efa0$uniquenesses

# 10. Bootstrap for Tucker’s φ & Hancock’s H (using hetcor from polycor)
B         <- 1000
phis      <- matrix(NA, B, k)
Hs        <- matrix(NA, B, k)
completed <- 0

while (completed < B) {
  # 10.1 Resample rows and select pruned columns
  samp <- df_mix2_clean[sample(nrow(df_mix2_clean), replace = TRUE), keep, drop = FALSE]
  
  # 10.2 Compute mixed correlations via hetcor
  het_out_b <- hetcor(
    as.data.frame(samp),
    use = "pairwise.complete.obs"
  )
  Rb <- het_out_b$correlations
  
  # 10.3 Fit MINRES + oblimin to the bootstrap correlation matrix
  fa_b <- tryCatch(
    fa(Rb,
       nfactors = k,
       fm       = "minres",
       rotate   = "oblimin",
       n.obs    = nrow(samp)),
    error = function(e) NULL
  )
  if (is.null(fa_b)) next
  
  # 10.4 Record congruence (Tucker’s φ) and H-index
  completed       <- completed + 1
  Lb              <- fa_b$loadings
  psib            <- fa_b$uniquenesses
  phis[completed, ] <- diag(factor.congruence(Lambda0, Lb))
  Hs[completed, ]   <- vapply(seq_len(k), function(j) {
    sum(Lb[, j])^2 / (sum(Lb[, j])^2 + sum(psib))
  }, numeric(1))
}

# 11. Report stability
cat("Mean Tucker’s φ: ", colMeans(phis), "\n")
cat("Mean Hancock’s H:", colMeans(Hs),   "\n")
