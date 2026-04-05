#!/usr/bin/env bash
set -euo pipefail

# If Conda is available in this shell, activate the expected environment.
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate Flogicity
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -n "${CONDA_PREFIX:-}" && -f "${CONDA_PREFIX}/python.exe" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/python.exe"
elif [[ -f "/mnt/d/Anaconda/envs/Flogicity/python.exe" ]]; then
  PYTHON_BIN="/mnt/d/Anaconda/envs/Flogicity/python.exe"
elif [[ -f "/mnt/d/Anaconda/python.exe" ]]; then
  PYTHON_BIN="/mnt/d/Anaconda/python.exe"
elif [[ -f "/d/Anaconda/envs/Flogicity/python.exe" ]]; then
  PYTHON_BIN="/d/Anaconda/envs/Flogicity/python.exe"
elif [[ -f "/d/Anaconda/python.exe" ]]; then
  PYTHON_BIN="/d/Anaconda/python.exe"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "No Python interpreter found. Set PYTHON_BIN to your Python executable." >&2
  exit 1
fi

echo "Using Python: $PYTHON_BIN"

EXPNAME="expert_100"
MAXSTEP=100

for s in 0; do
  "$PYTHON_BIN" main.py --config "config/tasks/sim/expert.yaml" \
    --exp "${EXPNAME}_${s}" --max-steps "$MAXSTEP" --seed "$s" \
    --log_dir log_sim
done
