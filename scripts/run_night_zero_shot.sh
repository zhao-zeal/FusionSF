#!/usr/bin/env bash
set -euo pipefail

# Run the six MMSP zero-shot experiments sequentially:
#   1. legacy, all VQ off
#   2. legacy, paper-best VQ
#   3. fixed_v1, all VQ off
#   4. fixed_v1, paper-best VQ
#   5. Chronos-2, power only
#   6. Chronos-2, power plus future NWP
#
# Usage:
#   bash scripts/run_night_zero_shot.sh [PHYSICAL_GPU_ID]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:-2}"
CHRONOS_SOURCE_DIR="${ROOT}/outputs/pipeline_v1_fixed/20260731_224035_fusionsf_fixedv1_clean30_zeroshot_train10_19_test0_9_seed42"

if [[ ! -d "${CHRONOS_SOURCE_DIR}" ]]; then
    echo "Chronos zero-shot source directory is unavailable: ${CHRONOS_SOURCE_DIR}" >&2
    exit 2
fi

cd "${ROOT}"

echo "[1/6] legacy zero-shot, all VQ off (GPU ${GPU_ID})"
bash scripts/run_night02z_legacy_zeroshot_vq_all_off.sh "${GPU_ID}"

echo "[2/6] legacy zero-shot, paper-best VQ (GPU ${GPU_ID})"
bash scripts/run_night01z_legacy_zeroshot_vq_paper_best.sh "${GPU_ID}"

echo "[3/6] fixed_v1 zero-shot, all VQ off (GPU ${GPU_ID})"
bash scripts/run_night03c_fixedv1_zeroshot_vq_all_off.sh "${GPU_ID}"

echo "[4/6] fixed_v1 zero-shot, paper-best VQ (GPU ${GPU_ID})"
bash scripts/run_night03d_fixedv1_zeroshot_vq_paper_best.sh "${GPU_ID}"

echo "[5/6] Chronos-2 zero-shot, power only (GPU ${GPU_ID})"
bash scripts/run_night04_chronos2_mmsp_power_baseline.sh \
    "${GPU_ID}" "${CHRONOS_SOURCE_DIR}"

echo "[6/6] Chronos-2 zero-shot, power plus future NWP (GPU ${GPU_ID})"
bash scripts/run_night04b_chronos2_mmsp_future_nwp.sh \
    "${GPU_ID}" "${CHRONOS_SOURCE_DIR}"

echo "All six zero-shot experiments completed."
