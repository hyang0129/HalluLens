#!/bin/bash -l

#SBATCH --job-name=jupyter_empire_PORT		# renamed to jupyter_empire_<port> below
#SBATCH --comment="Jupyter Lab + cell worker on Empire AI"

#SBATCH --account=rit
#SBATCH --partition=rit

#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

#SBATCH --mail-user=hy3134@rit.edu
#SBATCH --mail-type=END

#SBATCH --time=3-00:00:00			# overridden by launch_jupyter.py SBATCH_FLAGS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16			# overridden by launch_jupyter.py SBATCH_FLAGS
#SBATCH --mem-per-cpu=6g			# overridden by launch_jupyter.py SBATCH_FLAGS
#SBATCH --gres=gpu:1
#
# jupyter_with_worker.sbatch — a Jupyter Lab allocation that ALSO drains a
# cell-worker dispatch queue on launch. Submitted by scripts/launch_jupyter.py
# when --dispatch-root is given (so the guarded launcher's caps still apply).
#
# Behaviour: the cell worker is started in the background FIRST (in the base
# env, before `conda activate ml`, so the ml env's CUDA libs never leak into
# the worker's pinned p311 python). The worker drains the queue and exits when
# it is empty; Jupyter Lab then keeps running in the foreground, holding the
# allocation so the node stays accessible as normal — during and after the run.
#
# Positional args (passed by launch_jupyter.py --dispatch-root):
#   $1 = PORT
#   $2 = DISPATCH_ROOT  (absolute) — cell queue to drain (empty => no worker)
#   $3 = REPO_ROOT      (absolute) — HalluLens checkout holding worker_79.sh

PORT=${1:-8887}
DISPATCH_ROOT_ARG="${2:-}"
REPO_ROOT_ARG="${3:-}"

# Rename so launch_jupyter.py's cap-counting + port-collision logic sees this
# as a jupyter allocation (identical to empire_jupyter_lab.sh).
scontrol update JobId=${SLURM_JOB_ID} JobName=jupyter_empire_${PORT} 2>/dev/null || true

echo "Jupyter+worker on Empire AI (port=${PORT}, dispatch=${DISPATCH_ROOT_ARG:-none})"

# --- 1. Start the cell worker in the background (clean base env) -------------
if [ -n "$DISPATCH_ROOT_ARG" ] && [ -n "$REPO_ROOT_ARG" ]; then
  WORKER_LOG="${REPO_ROOT_ARG}/shared/jupyter_worker_${PORT}_${SLURM_JOB_ID}.log"
  echo "Starting cell worker (drains ${DISPATCH_ROOT_ARG}) -> ${WORKER_LOG}"
  # run_experiment.py writes outputs relative to cwd — anchor at the repo so they
  # land in <repo>/runs/... where worker_79's output_check looks (not $HOME/runs).
  cd "$REPO_ROOT_ARG"
  PROJECT_ROOT="$REPO_ROOT_ARG" \
  DISPATCH_ROOT="$DISPATCH_ROOT_ARG" \
  DISPATCH_NODE="$(hostname)-${PORT}" \
  nohup bash "${REPO_ROOT_ARG}/scripts/dispatch/worker_79.sh" > "$WORKER_LOG" 2>&1 &
  echo "cell worker PID $! (logs: ${WORKER_LOG})"
else
  echo "No --dispatch-root given — running as a plain Jupyter allocation."
fi

# --- 2. Start Jupyter Lab in the foreground (holds the allocation) ----------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

echo "Launching Jupyter Lab on port ${PORT}"
jupyter lab --ip=0.0.0.0 --no-browser --port=${PORT}
