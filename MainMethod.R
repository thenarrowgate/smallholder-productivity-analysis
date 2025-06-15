# load requires packages
library(psych)
library(EFA.dimensions)
library(EFAutilities)
library(mgcv)
library(lavaan)
library(semTools)
library(dplyr)
set.seed(2025)

# load data
setwd("E:/Atuda/67814-Data-Science-Final-Project/Code")
df_all   <- readxl::read_excel("nepal_dataframe_FA.xlsx", sheet = 1)

# extract outcome and predictors
y_prod <- df_all$Q0__AGR_PROD__continuous
df_all <- df_all %>% select(-Q0__AGR_PROD__continuous, -Q0__sustainable_livelihood_score__continuous)

# Identify numeric vs nominal features via your parse_feature_metadata()
library(stringr)
type_from_name <- function(name) {
  # simple regex to pull TYPE_PATTERN out of column name
  str_match(name, "__(continuous|ordinal|binary|nominal|time)")[,2]
}
col_types <- sapply(names(df_all), type_from_name)
df_num     <- df_all[, col_types %in% c("continuous","ordinal","binary")]
df_nominal <- df_all[, which(col_types == "nominal")]

# Compute Spearman R
R <- cor(df_num, method = "spearman", use = "pairwise.complete.obs")

# Parallel analysis (minres factors) with 500 iterations, 95% quantile
fa.parallel(R,
            n.obs    = nrow(df_num),
            fm       = "minres",
            fa       = "fa",
            n.iter   = 500,
            quant    = .95,
            cor      = "spearman",
            plot     = FALSE) -> pa_out

k_PA <- pa_out$nfact    # suggested upper bound :contentReference[oaicite:6]{index=6}

# MAP on the same R (spearman), providing Ncases
map_res <- MAP(data    = df_num,
               corkind = "spearman",
               Ncases  = nrow(df_num),
               verbose = FALSE)

# now extract the suggested number of factors
k_MAP <- map_res$NfactorsMAP

