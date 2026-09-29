#!/bin/bash
# Build the pinned experiment environment in WSL2 (native ext4, not /mnt/c).
# Layout:
#   ~/ova/venv            pinned Python 3.11 virtualenv
#   ~/ova/src/robomimic   robomimic checkout at tag v0.5.0
#   ~/ova/datasets        pristine + working HDF5 files (read-only pristine copies)
#   ~/ova/runs            per-run output directories (large; not in git)
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"   # uv lives here; not on PATH in non-login shells
OVA="$HOME/ova"
mkdir -p "$OVA/datasets" "$OVA/runs" "$OVA/src" "$OVA/logs"
cd "$OVA/src"
if [ ! -f robomimic/setup.py ]; then
  rm -rf robomimic
  for attempt in 1 2 3 4 5; do
    git clone --branch v0.5.0 --depth 1 https://github.com/ARISE-Initiative/robomimic.git && break
    echo "clone attempt $attempt failed; retrying in 15s"; rm -rf robomimic; sleep 15
  done
fi
cd robomimic && echo "robomimic commit: $(git rev-parse HEAD)" && cd ..

if [ ! -x "$OVA/venv/bin/python" ]; then
  uv venv --python 3.11 "$OVA/venv"
fi
source "$OVA/venv/bin/activate"

# ---- pins (input) ----
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchvision==0.23.0
uv pip install \
  numpy==1.26.4 \
  mujoco==3.2.6 \
  robosuite==1.5.1 \
  h5py==3.11.0 \
  scipy==1.13.1 \
  pandas==2.2.2 \
  statsmodels==0.14.2 \
  matplotlib==3.9.2 \
  tensorboard==2.17.1 tensorboardX==2.6.2.2 \
  imageio==2.35.1 imageio-ffmpeg==0.5.1 \
  psutil tqdm termcolor egl_probe \
  "huggingface_hub==0.23.4" "transformers==4.41.2" "diffusers==0.11.1"
uv pip install --no-deps -e "$OVA/src/robomimic"

# ---- lock (output) ----
uv pip freeze > "$OVA/requirements-lock.txt"
echo "lock written: $OVA/requirements-lock.txt"

# ---- sanity ----
python - <<'EOF'
import torch, numpy, mujoco, robosuite, robomimic, h5py, statsmodels
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.version.cuda)
print("numpy", numpy.__version__, "mujoco", mujoco.__version__, "robosuite", robosuite.__version__,
      "robomimic", robomimic.__version__, "h5py", h5py.__version__, "statsmodels", statsmodels.__version__)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
EOF
echo "ENV_SETUP_DONE"
