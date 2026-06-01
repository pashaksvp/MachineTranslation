#!/usr/bin/env bash
set -euo pipefail

COMPETITION="deep-past-initiative-machine-translation"
OUT_DIR="${1:-data/raw}"
FILES=("train.csv" "test.csv" "sample_submission.csv")

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

for FILE in "${FILES[@]}"; do
  "${KAGGLE_BIN}" competitions download -c "${COMPETITION}" -f "${FILE}" -p "${OUT_DIR}"
  ARCHIVE="${OUT_DIR}/${FILE}.zip"
  if [ -f "${ARCHIVE}" ]; then
    unzip -o "${ARCHIVE}" -d "${OUT_DIR}"
  fi
done

echo "Downloaded files:"
find "${OUT_DIR}" -maxdepth 1 -type f -print
