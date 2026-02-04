#!/bin/bash
# RW-GCN 실험 스크립트
# 모든 데이터셋에 대해 R-GCN baseline과 RW-GCN을 비교

echo "============================================================"
echo "RW-GCN Experiments: R-GCN vs RW-GCN"
echo "============================================================"

DATASETS=("AIFB" "MUTAG" "BGS" "AM")
RUNS=10
EPOCHS=50

# 결과 저장
echo ""
echo "| Model | AIFB | MUTAG | BGS | AM |" > results.md
echo "|-------|------|-------|-----|-----|" >> results.md

# R-GCN Baseline 실험
echo ""
echo ">>> Running R-GCN Baseline..."
RGCN_RESULTS=""
for dataset in "${DATASETS[@]}"; do
    echo "  - Dataset: $dataset"
    result=$(python rgcn_baseline.py --dataset $dataset --runs $RUNS --epochs $EPOCHS 2>&1 | grep "Test Accuracy:" | tail -1)
    acc=$(echo $result | grep -oP '\d+\.\d+ ± \d+\.\d+')
    RGCN_RESULTS="$RGCN_RESULTS $acc |"
    echo "    Result: $acc"
done
echo "| R-GCN (ours) |$RGCN_RESULTS" >> results.md

# RW-GCN 실험 (no gate)
echo ""
echo ">>> Running RW-GCN (no gate)..."
RWGCN_RESULTS=""
for dataset in "${DATASETS[@]}"; do
    echo "  - Dataset: $dataset"
    result=$(python rwgcn.py --dataset $dataset --k_hop 3 --learnable_scale --runs $RUNS --epochs $EPOCHS 2>&1 | grep "Test Accuracy:" | tail -1)
    acc=$(echo $result | grep -oP '\d+\.\d+ ± \d+\.\d+')
    RWGCN_RESULTS="$RWGCN_RESULTS $acc |"
    echo "    Result: $acc"
done
echo "| RW-GCN |$RWGCN_RESULTS" >> results.md

# RW-GCN 실험 (with gate)
echo ""
echo ">>> Running RW-GCN + Gate..."
RWGCN_GATE_RESULTS=""
for dataset in "${DATASETS[@]}"; do
    echo "  - Dataset: $dataset"
    result=$(python rwgcn.py --dataset $dataset --k_hop 3 --learnable_scale --use_gate --runs $RUNS --epochs $EPOCHS 2>&1 | grep "Test Accuracy:" | tail -1)
    acc=$(echo $result | grep -oP '\d+\.\d+ ± \d+\.\d+')
    RWGCN_GATE_RESULTS="$RWGCN_GATE_RESULTS $acc |"
    echo "    Result: $acc"
done
echo "| RW-GCN + Gate |$RWGCN_GATE_RESULTS" >> results.md

# 논문 참조값 추가
echo "| R-GCN (paper) | 95.83 | 73.23 | 83.10 | 89.29 |" >> results.md

echo ""
echo "============================================================"
echo "Experiments Complete! Results saved to results.md"
echo "============================================================"
cat results.md