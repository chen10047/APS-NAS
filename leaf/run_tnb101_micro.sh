#!/bin/bash

export PYTHONHASHSEED=0
SEARCH_SPACE="transnb101_micro"
Lambda=0.3
DATASETS=("class_scene" "class_object" "autoencoder" "jigsaw")   # 需要遍历的数据集
#BASE_OUTPUT_DIR="./exp1000/tree4_23_no_space_tnb101_micro"
BASE_OUTPUT_DIR="./exp_nsga_fused_leaf_final/depth6_pop100_lambda${Lambda}/tnb101_micro"
SEEDS=(1 2 3 4 5)
#SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)

mkdir -p ${BASE_OUTPUT_DIR}

for DATASET in "${DATASETS[@]}"
do
    echo "=========================================="
    echo "Starting experiments for dataset: ${DATASET}"
    echo "=========================================="

    OUTPUT_BASE_DIR="${BASE_OUTPUT_DIR}/${DATASET}"
    mkdir -p ${OUTPUT_BASE_DIR}

    for SEED in "${SEEDS[@]}"
    do
        echo "----------------------------------------"
        echo "Dataset: ${DATASET} | Seed: ${SEED}"
        echo "----------------------------------------"

        START_TIME=$(date +%s)

        python ./leaf/run_nsga2_tnb101_micro.py \
            --search_space ${SEARCH_SPACE} \
            --dataset ${DATASET} \
            --seed ${SEED} \
            --lambda ${Lambda} \
            --output_dir ${OUTPUT_BASE_DIR}

        echo "Finished seed ${SEED} for dataset ${DATASET} (took ${DURATION} seconds)"
        echo ""
    done

    echo "All seeds completed for dataset: ${DATASET}"
done

echo "All datasets processed. Results saved in ${BASE_OUTPUT_DIR}"
echo "Time logs saved in ${BASE_OUTPUT_DIR}/time_*.csv"
