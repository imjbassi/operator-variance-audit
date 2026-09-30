#!/bin/bash
# Launch NW queue workers under nohup inside WSL. Idempotent: the queue skips completed runs.
# Usage: ARMS="A B C" NW=3 bash env/launch_workers.sh
set -u
ARMS=${ARMS:-A B C}
NW=${NW:-3}
EPOCHS=${EPOCHS:-2000}
REPO=${REPO:-/mnt/c/Users/jaive.DESKTOP-3TNM9JL/Desktop/operator-variance-audit}
cd "$REPO"
source ~/ova/venv/bin/activate
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
mkdir -p ~/ova/logs
if pgrep -f 'scripts/run_queue.py' > /dev/null; then
  echo "workers already running; not launching"; exit 0
fi
# No worker alive => every remaining lock is stale (left by a killed worker). Clear them.
find ~/ova/runs -name .lock -delete 2>/dev/null
stamp=$(date +%Y%m%d_%H%M%S)
for w in $(seq 0 $((NW-1))); do
  nohup python scripts/run_queue.py --arms $ARMS --worker $w --epochs $EPOCHS \
     > ~/ova/logs/worker_${stamp}_$w.log 2>&1 &
  echo "$(date '+%F %T') launched worker $w pid $!"
  sleep 20   # stagger so workers do not claim the same run at the same instant
done
