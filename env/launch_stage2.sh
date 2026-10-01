#!/bin/bash
# Launch NW Stage 2 evaluation workers under nohup inside WSL (Amendment 2). Idempotent.
# Usage: NW=8 bash env/launch_stage2.sh
set -u
NW=${NW:-8}
REPO=${REPO:-/mnt/c/Users/jaive.DESKTOP-3TNM9JL/Desktop/operator-variance-audit}
cd "$REPO"
source ~/ova/venv/bin/activate
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
# one BLAS/torch thread per worker: the rollouts are CPU-bound single-thread MuJoCo + tiny LSTM
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p ~/ova/logs
if pgrep -f 'scripts/eval_stage2_queue.py' > /dev/null; then
  echo "stage-2 workers already running; not launching"; exit 0
fi
# No worker alive => every remaining chunk lock is stale. Clear them.
find ~/ova/runs -name '.lock_*' -path '*/eval_stage2/*' -delete 2>/dev/null
stamp=$(date +%Y%m%d_%H%M%S)
for w in $(seq 0 $((NW-1))); do
  nohup python scripts/eval_stage2_queue.py --worker $w > ~/ova/logs/s2_worker_${stamp}_$w.log 2>&1 &
  echo "$(date '+%F %T') launched stage-2 worker $w pid $!"
  sleep 3
done
