#!/usr/bin/env bash
set -euo pipefail

COMPETITION="deep-past-initiative-machine-translation"
OUT_DIR="${1:-data/raw}"

mkdir -p "${OUT_DIR}"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Kaggle CLI is not installed."
  echo "Install it with: pip install kaggle"
  echo "Then configure ~/.kaggle/kaggle.json from your Kaggle account settings."
  exit 1
fi

kaggle competitions download -c "${COMPETITION}" -p "${OUT_DIR}"

ARCHIVE="${OUT_DIR}/${COMPETITION}.zip"
if [ -f "${ARCHIVE}" ]; then
  unzip -o "${ARCHIVE}" -d "${OUT_DIR}"
fi

echo "Downloaded files:"
find "${OUT_DIR}" -maxdepth 1 -type f -print
