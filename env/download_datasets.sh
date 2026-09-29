#!/bin/bash
# Download pristine robomimic v1.5 Multi-Human low-dim datasets (lift, can, square)
# into ~/ova/datasets/pristine and make them read-only. Records sha256 sums.
set -euo pipefail
OVA="$HOME/ova"
DST="$OVA/datasets/pristine"
mkdir -p "$DST"
# robomimic v0.5.0 download script (registry serves v1.5 files from the Hugging Face mirror)
ROBOMIMIC_SRC="${ROBOMIMIC_SRC:-$OVA/src/robomimic}"
PY="${PY:-$OVA/venv/bin/python}"
"$PY" "$ROBOMIMIC_SRC/robomimic/scripts/download_datasets.py" \
  --tasks lift can square --dataset_types mh --hdf5_types low_dim \
  --download_dir "$DST"
find "$DST" -name '*.hdf5' -exec chmod a-w {} \;
( cd "$DST" && find . -name '*.hdf5' | sort | xargs sha256sum ) | tee "$DST/SHA256SUMS"
echo "DOWNLOAD_DONE"
