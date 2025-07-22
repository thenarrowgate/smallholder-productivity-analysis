# Smallholder Productivity Analysis

This repository analyzes smallholder farming data from **Nepal** and **Senegal** to uncover latent factors that drive productivity. The workflow combines Python preprocessing with extensive factor analysis in R.

## Project Overview
- **Data sources:** Raw survey spreadsheets are located in `data/raw` with cleaned variable names in `data/raw_updated_names`.
- **Preprocessing:** The notebook `EDA.ipynb` performs exploratory analysis and handles missing values and outliers. It saves processed datasets as `nepal_dataframe_FA.xlsx` and `senegal_dataframe_FA.xlsx`.
- **Factor analysis:** `MainMethod.R` splits variables by type, builds mixed correlation matrices and uses bootstrapped exploratory factor analysis (EFA) to obtain stable loadings. Further steps (e.g., CFA/SEM) examine how well latent factors explain the productivity index.
Before EFA, the script now checks variable suitability using the KMO and Bartlett tests. The correlation matrix is adjusted to be positive definite using nearPD before these tests.
## Environment Setup
1. **Python requirements**
   ```bash
   pip install -r requirements.txt
   ```
   or create the conda environment
   ```bash
   conda env create -f environment.yml
   ```
2. **R setup** – install system and R dependencies
   ```bash
   ./.setup/setup.sh
   ```
   This script reads `apt.txt` (which includes `r-base`) and installs
   the EFAtools and Gifi packages.

## Running the Analysis
- **Run preprocessing**
  ```bash
  jupyter notebook EDA.ipynb
  ```
  The notebook will save cleaned datasets used by the R script.
- **Run factor analysis**
  Ensure `.setup/setup.sh` has been executed to install dependencies.
  Update `LOCAL_DIR` in `MainMethod.R` to the repository path and then execute:
  ```bash
  Rscript MainMethod.R
  ```

The goal is to obtain a reliable factor solution describing productivity. See `Agents.md` for additional project details.
