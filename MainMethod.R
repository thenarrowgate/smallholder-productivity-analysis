# load requires packages
library(readxl)    # Excel I/O
library(dplyr)     # Data wrangling
library(stringr)   # Regex utilities
library(EFAtools)  # EFA retention criteria, PARALLEL, VSS
library(psych)     # fa.parallel, fa(), factor.congruence()
library(boot)      # bootstrap()
library(mgcv)      # gam()
library(lavaan)    # sem(), fitMeasures()
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


# calculate spearman corr matrix
R <- cor(nepal$df_num, method="spearman", use="pairwise.complete.obs")

# get reliable bracket for k
pa <- fa.parallel(R, n.obs = n,
                  fm = "minres", fa = "fa",
                  n.iter = 500, quant = .95,
                  plot = FALSE) 
kPA <- pa$nfact

vss_res <- VSS(R, n = ncol(R), fm = "minres", n.obs = n, plot = FALSE)
kMAP <- which.min(vss_res$map)

k <- kPA  # or choose within [kMAP,kPA]

R0      <- cor(nepal$df_num, method="spearman")
efa0    <- fa(R0, nfactors = k, fm = "minres", rotate = "oblimin")
Lambda0 <- efa0$loadings

