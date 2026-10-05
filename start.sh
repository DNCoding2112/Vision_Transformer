#!/bin/bash
#SBATCH --job-name=amdt_train
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=23:00:00

# ============================================================
# Logging
# ============================================================
LOGFILE="run_${SLURM_JOB_ID:-manual}.log"
exec > >(tee -a $LOGFILE) 2>&1

echo "===================================="
echo "JOB START"
date
echo "SLURM_JOB_ID       : $SLURM_JOB_ID"
echo "SLURM_NODELIST     : $SLURM_NODELIST"
echo "SLURM_CPUS_ON_NODE : $SLURM_CPUS_ON_NODE"
echo "Working directory  : $(pwd)"
echo "===================================="

# ============================================================
# Environment
# ============================================================
module load python/3.11.14

# Persistent env in home dir — survives job restarts
ENV_DIR="$HOME/amdt_env"

if [ ! -d "$ENV_DIR" ]; then
    echo "First run: creating environment at $ENV_DIR"
    uv venv "$ENV_DIR"
    source "$ENV_DIR/bin/activate"

    # Match cluster CUDA version — check nvidia-smi output above and adjust
    # cu118 = CUDA 11.8, cu121 = CUDA 12.1, cu124 = CUDA 12.4
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    uv pip install -r requirements.txt
else
    echo "Reusing existing environment at $ENV_DIR"
    source "$ENV_DIR/bin/activate"
fi

echo "Python version : $(python --version)"
echo "uv version     : $(uv --version)"

# ============================================================
# Sanity checks
# ============================================================
echo "===================================="
echo "GPU / CUDA INFO"
nvidia-smi
echo "===================================="
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}:', torch.cuda.get_device_name(i))
"

# ============================================================
# Datasets — only prepare if not already done
# ============================================================
if [ ! -d "data/cifake/train/REAL" ] \
   || [ ! -d "data/aivsreal/train/REAL" ] \
   || [ ! -d "data/faces140k/train/REAL" ]; then
    echo "Preparing datasets (first run)..."
    python download_datasets.py
else
    echo "Datasets already prepared — skipping download/prepare."
fi

# ============================================================
# Smoke test — cheap, catches 90% of pipeline bugs
# ============================================================
echo "===================================="
echo "Running smoke test..."
python main.py --test
if [ $? -ne 0 ]; then
    echo "SMOKE TEST FAILED — aborting before wasting GPU time."
    exit 1
fi
echo "Smoke test passed."
echo "===================================="

# ============================================================
# Training — torchrun for 2-GPU DDP
# ============================================================
echo "Starting training on 2 GPUs with torchrun..."
# torchrun --nproc_per_node=2 --master_port=29500 main.py --mode train
if [ -f "$HOME/sem7/outputs/checkpoints/amdt_resume.pt" ]; then
    echo "Resume checkpoint found — continuing from last epoch"
    torchrun --nproc_per_node=2 --master_port=29500 main.py --mode train --resume
else
    echo "No resume checkpoint — starting fresh"
    torchrun --nproc_per_node=2 --master_port=29500 main.py --mode train
fi

EXIT_CODE=$?

echo "===================================="
echo "JOB FINISHED — exit code $EXIT_CODE"
date
echo "===================================="
exit $EXIT_CODE