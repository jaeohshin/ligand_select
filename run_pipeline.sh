#!/bin/bash
# run_pipeline.sh
# Run ligand selection pipeline for a target.
# Usage: bash run_pipeline.sh --config targets/T2383/config.yaml
#        bash run_pipeline.sh --config targets/T2383/config.yaml --from-step 3 --to-step 5

set -e  # stop on error

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FROM_STEP=0
TO_STEP=5
CONFIG=""

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)      CONFIG="$2";     shift 2 ;;
        --from-step)   FROM_STEP="$2";  shift 2 ;;
        --to-step)     TO_STEP="$2";    shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Usage: bash run_pipeline.sh --config <config.yaml> [--from-step N] [--to-step N]"
    exit 1
fi

echo "============================================"
echo "  CASP17 Ligand Selection Pipeline"
echo "  Config:     $CONFIG"
echo "  Steps:      $FROM_STEP → $TO_STEP"
echo "============================================"

run_step() {
    local step=$1
    local name=$2
    local cmd=$3

    if [[ $step -lt $FROM_STEP || $step -gt $TO_STEP ]]; then
        return 0
    fi

    echo ""
    echo "--------------------------------------------"
    echo "  Step $step: $name"
    echo "--------------------------------------------"
    eval "$cmd"
    echo "  Step $step done."
}

run_step 0 "Build index (0a)" \
    "python ligand_select/steps/step0a_build_index.py --config $CONFIG"

run_step 0 "Conformation PCA (0b)" \
    "python ligand_select/steps/step0b_conformation_pca.py --config $CONFIG"

run_step 1 "Cluster pockets" \
    "python ligand_select/steps/step1_cluster_pockets.py --config $CONFIG"

run_step 2 "Validate pockets (P2Rank)" \
    "python ligand_select/steps/step2_validate_pockets.py --config $CONFIG"

run_step 3 "Cluster poses" \
    "python ligand_select/steps/step3_cluster_poses.py --config $CONFIG"

run_step 4 "Score poses (GNINA)" \
    "python ligand_select/steps/step4_score_poses.py --config $CONFIG"

run_step 5 "Final selection report" \
    "python ligand_select/steps/step5_select_final.py --config $CONFIG"

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "============================================"
