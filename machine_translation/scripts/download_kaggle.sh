#!/usr/bin/env bash
set -euo pipefail

COMPETITION="deep-past-initiative-machine-translation"
OUT_DIR="${1:-data/raw}"

mkdir -p "${OUT_DIR}"

if [ -x ".venv/bin/kaggle" ]; then
  KAGGLE_BIN=".venv/bin/kaggle"
elif command -v kaggle >/dev/null 2>&1; then
  KAGGLE_BIN="kaggle"
else
  echo "Kaggle CLI is not installed."
  echo "Install project dependencies with: pip install -e \".[dev]\""
  echo "Then authenticate with: kaggle auth login"
  exit 1
fi

"${KAGGLE_BIN}" competitions download -c "${COMPETITION}" -p "${OUT_DIR}"

ARCHIVE="${OUT_DIR}/${COMPETITION}.zip"
if [ -f "${ARCHIVE}" ]; then
  unzip -o "${ARCHIVE}" -d "${OUT_DIR}"
fi

echo "Downloaded files:"
find "${OUT_DIR}" -maxdepth 1 -type f -print
