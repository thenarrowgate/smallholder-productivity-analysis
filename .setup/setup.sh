#!/usr/bin/env bash
set -e

# Install R packages required by the analysis
"$(dirname "$0")"/install_r_packages.sh
