#!/bin/bash
set -e

# ============================================================
# FlashOverlap full pipeline: profile → bandwidth → search → test
# Usage:
#   bash run.sh                                          # defaults: M=4096 N=4096 K=4096, all_reduce, 8 GPUs
#   bash run.sh 4096 8192 4096                           # all_reduce, 8 GPUs
#   bash run.sh 4096 8192 4096 all_reduce                # 8 GPUs
#   bash run.sh 4096 8192 4096 all_reduce 0,1            # 2 GPUs
#   bash run.sh 4096 8192 4096 reduce_scatter 0,1,2,3    # 4 GPUs
#
# Set FLASH_DEBUG=1 to enable verbose debug output:
#   FLASH_DEBUG=1 bash run.sh 4096 4096 4096
# ============================================================

M=${1:-4096}
N=${2:-4096}
K=${3:-4096}
COMM_OP=${4:-"all_reduce"}
DEVICES=${5:-"0,1,2,3,4,5,6,7"}

# Count GPUs from comma-separated device list
NPROC=$(echo "$DEVICES" | awk -F',' '{print NF}')

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo " FlashOverlap Pipeline"
echo " M=$M  N=$N  K=$K"
echo " DEVICES=$DEVICES  NPROC=$NPROC"
echo " COMM_OP=$COMM_OP"
echo "============================================"

# Ensure configs dir exists
mkdir -p "$SCRIPT_DIR/configs"

# Step 1: Profile GEMM configs (single GPU)
echo ""
echo "[Step 1/4] Profiling GEMM configs..."
cd "$SCRIPT_DIR/tune"
MUSA_VISIBLE_DEVICES="${DEVICES%%,*}" python3 profile_config.py \
    --m "$M" --n "$N" --k "$K"

# Step 2: Generate bandwidth curve
echo ""
echo "[Step 2/4] Generating bandwidth curve..."
cd "$SCRIPT_DIR/tune"
MUSA_VISIBLE_DEVICES="$DEVICES" torchrun --nproc_per_node="$NPROC" \
    bandwidth.py --comm_op "$COMM_OP"

# Step 3: Search for wave group size
echo ""
echo "[Step 3/4] Searching for wave group size..."
cd "$SCRIPT_DIR/tune"
MUSA_VISIBLE_DEVICES="$DEVICES" torchrun --nproc_per_node="$NPROC" \
    search.py --m_dim "$M" --n_dim "$N" --k_dim "$K" \
    --comm_op "$COMM_OP" --predictive_search

# Step 4: Speed test
echo ""
echo "[Step 4/4] Running speed test..."
cd "$SCRIPT_DIR/test"
MUSA_VISIBLE_DEVICES="$DEVICES" torchrun --nproc_per_node="$NPROC" \
    test.py --m_dim "$M" --n_dim "$N" --k_dim "$K" \
    --comm_op "$COMM_OP"

echo ""
echo "============================================"
echo " Pipeline complete!"
echo "============================================"
