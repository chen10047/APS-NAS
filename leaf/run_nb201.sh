#!/bin/bash

export PYTHONHASHSEED=0
SEARCH_SPACE="nb201"
Lambda=0.3
DATASETS=("cifar10" "cifar100" "ImageNet16-120")   # 需要遍历的数据集
#BASE_OUTPUT_DIR="./exp1000/tree4_no_space_nb201"
BASE_OUTPUT_DIR="./exp_nsga_fused_leaf_final/depth5_pop100_lambda${Lambda}/nb201"
#SEEDS=(1 2 3 4 5 6 7 8 9 10 )
SEEDS=(11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)
#
# 创建基础输出目录（如果不存在）
#mkdir -p ${BASE_OUTPUT_DIR}

# 遍历每个数据集
for DATASET in "${DATASETS[@]}"
do
    echo "=========================================="
    echo "Starting experiments for dataset: ${DATASET}"
    echo "=========================================="

    # 为当前数据集创建独立的输出子目录
    OUTPUT_BASE_DIR="${BASE_OUTPUT_DIR}/${DATASET}"
    mkdir -p ${OUTPUT_BASE_DIR}

#     遍历所有随机种子
    for SEED in "${SEEDS[@]}"
    do
        echo "----------------------------------------"
        echo "Dataset: ${DATASET} | Seed: ${SEED}"
        echo "----------------------------------------"
        python ./leaf/run_nsga2_nb201.py \
            --search_space ${SEARCH_SPACE} \
            --dataset ${DATASET} \
            --seed ${SEED} \
            --lambda ${Lambda} \
            --output_dir ${OUTPUT_BASE_DIR}
        echo "Finished seed ${SEED} for dataset ${DATASET}"
        echo ""
    done

    echo "All seeds completed for dataset: ${DATASET}"

done

echo "All datasets processed. Results saved in ${BASE_OUTPUT_DIR}"
