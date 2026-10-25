#!/bin/bash
# 运行多个随机种子的 NSGA-II 遗传编程实验
export PYTHONHASHSEED=0
SEARCH_SPACE="nb101"
DATASET="cifar10"
Lambda=0.3
#OUTPUT_BASE_DIR="./exp_subtree/tree4_23_no_space_nb101"
OUTPUT_BASE_DIR="./exp_nsga_fused_leaf_zen0/depth5_pop100_lambda${Lambda}/nb101"
SEEDS=(1 2 3 4 5 6 7 8 9 10)
#SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)

mkdir -p ${OUTPUT_BASE_DIR}

for SEED in "${SEEDS[@]}"
do
    echo "========================================"
    echo "Running with seed ${SEED}..."
    echo "========================================"
    python ./leaf/run_nsga2_nb101.py \
        --search_space ${SEARCH_SPACE} \
        --dataset ${DATASET} \
        --seed ${SEED} \
        --lambda ${Lambda} \
        --output_dir ${OUTPUT_BASE_DIR}
    echo "Finished seed ${SEED}"
    echo ""
done

echo "All runs completed. Results saved in ${OUTPUT_BASE_DIR}"
