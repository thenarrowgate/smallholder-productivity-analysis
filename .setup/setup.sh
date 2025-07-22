#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(dirname "$0")"
REPO_ROOT="${SCRIPT_DIR}/.."

# Install system packages listed in apt.txt if apt-get is available
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    grep -v '^#' "${REPO_ROOT}/apt.txt" | xargs sudo apt-get install -y
fi

# Install R packages required by the analysis
"${SCRIPT_DIR}/install_r_packages.sh"
