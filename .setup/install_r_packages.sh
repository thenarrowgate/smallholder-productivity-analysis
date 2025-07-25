#!/usr/bin/env bash

# Install required R packages for MainMethod.R
set -e

Rscript - <<'RSCRIPT'
packages <- c("EFAtools", "Gifi", "mgcViz", "mediation", "semTools", "blavaan")
install.packages(packages, repos = "https://cloud.r-project.org")
RSCRIPT
