#!/bin/bash

# Test script for CUDA recitation 8 exercise
# Run: ./test_steps.sh [step_number|all]

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if nvcc is available
check_nvcc() {
    if ! command -v nvcc &> /dev/null; then
        echo -e "${RED}ERROR: nvcc not found. Load CUDA module or check your PATH.${NC}"
        exit 1
    fi
}

test_step1() {
    echo -e "${YELLOW}Testing Step 1: VectorAdd${NC}"

    if [ ! -f step1_vectoradd.cu ]; then
        echo -e "${RED}FAIL: step1_vectoradd.cu not found${NC}"
        return 1
    fi

    nvcc -o step1 step1_vectoradd.cu 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}FAIL: Compilation failed${NC}"
        nvcc -o step1 step1_vectoradd.cu
        return 1
    fi

    output=$(./step1 2>&1)

    if echo "$output" | grep -q "PASSED"; then
        echo -e "${GREEN}PASS: VectorAdd correct${NC}"
        echo "$output"
        return 0
    else
        echo -e "${RED}FAIL: VectorAdd incorrect${NC}"
        echo "$output"
        return 1
    fi
}

test_step2() {
    echo -e "${YELLOW}Testing Step 2: Reduction Baseline${NC}"

    if [ ! -f step2_reduce_baseline.cu ]; then
        echo -e "${RED}FAIL: step2_reduce_baseline.cu not found${NC}"
        return 1
    fi

    nvcc -o step2 step2_reduce_baseline.cu 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}FAIL: Compilation failed${NC}"
        nvcc -o step2 step2_reduce_baseline.cu
        return 1
    fi

    output=$(./step2 2>&1)

    if echo "$output" | grep -q "PASSED"; then
        echo -e "${GREEN}PASS: Baseline reduction correct${NC}"
        echo "$output"
        return 0
    else
        echo -e "${RED}FAIL: Baseline reduction incorrect${NC}"
        echo "$output"
        return 1
    fi
}

test_step3() {
    echo -e "${YELLOW}Testing Step 3: Reduce Warp Divergence${NC}"

    if [ ! -f step3_reduce_divergence.cu ]; then
        echo -e "${RED}FAIL: step3_reduce_divergence.cu not found${NC}"
        return 1
    fi

    nvcc -o step3 step3_reduce_divergence.cu 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}FAIL: Compilation failed${NC}"
        nvcc -o step3 step3_reduce_divergence.cu
        return 1
    fi

    output=$(./step3 2>&1)

    if echo "$output" | grep -q "PASSED"; then
        echo -e "${GREEN}PASS: Divergence-free reduction correct${NC}"
        echo "$output"
        return 0
    else
        echo -e "${RED}FAIL: Divergence-free reduction incorrect${NC}"
        echo "$output"
        return 1
    fi
}

test_step4() {
    echo -e "${YELLOW}Testing Step 4: Fix Bank Conflicts${NC}"

    if [ ! -f step4_reduce_bankconflict.cu ]; then
        echo -e "${RED}FAIL: step4_reduce_bankconflict.cu not found${NC}"
        return 1
    fi

    nvcc -o step4 step4_reduce_bankconflict.cu 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}FAIL: Compilation failed${NC}"
        nvcc -o step4 step4_reduce_bankconflict.cu
        return 1
    fi

    output=$(./step4 2>&1)

    if echo "$output" | grep -q "PASSED"; then
        echo -e "${GREEN}PASS: Bank-conflict-free reduction correct${NC}"
        echo "$output"
        return 0
    else
        echo -e "${RED}FAIL: Bank-conflict-free reduction incorrect${NC}"
        echo "$output"
        return 1
    fi
}

test_step5() {
    echo -e "${YELLOW}Testing Step 5: Reduce Idle Threads${NC}"

    if [ ! -f step5_reduce_idle.cu ]; then
        echo -e "${RED}FAIL: step5_reduce_idle.cu not found${NC}"
        return 1
    fi

    nvcc -o step5 step5_reduce_idle.cu 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}FAIL: Compilation failed${NC}"
        nvcc -o step5 step5_reduce_idle.cu
        return 1
    fi

    output=$(./step5 2>&1)

    if echo "$output" | grep -q "PASSED"; then
        echo -e "${GREEN}PASS: Idle-thread-optimized reduction correct${NC}"
        echo "$output"
        return 0
    else
        echo -e "${RED}FAIL: Idle-thread-optimized reduction incorrect${NC}"
        echo "$output"
        return 1
    fi
}

test_step6() {
    echo -e "${YELLOW}Testing Step 6: Unroll Last Warp${NC}"

    if [ ! -f step6_reduce_unroll.cu ]; then
        echo -e "${RED}FAIL: step6_reduce_unroll.cu not found${NC}"
        return 1
    fi

    nvcc -o step6 step6_reduce_unroll.cu 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}FAIL: Compilation failed${NC}"
        nvcc -o step6 step6_reduce_unroll.cu
        return 1
    fi

    output=$(./step6 2>&1)

    if echo "$output" | grep -q "PASSED"; then
        echo -e "${GREEN}PASS: Warp-unrolled reduction correct${NC}"
        echo "$output"
        return 0
    else
        echo -e "${RED}FAIL: Warp-unrolled reduction incorrect${NC}"
        echo "$output"
        return 1
    fi
}

# Main
check_nvcc

if [ -z "$1" ]; then
    echo "Usage: ./test_steps.sh [1|2|3|4|5|6|all]"
    echo ""
    echo "  1   VectorAdd (CUDA basics)"
    echo "  2   Reduction baseline"
    echo "  3   Fix warp divergence"
    echo "  4   Fix bank conflicts"
    echo "  5   Reduce idle threads"
    echo "  6   Unroll last warp"
    echo "  all Run all steps"
    exit 0
fi

case $1 in
    1) test_step1 ;;
    2) test_step2 ;;
    3) test_step3 ;;
    4) test_step4 ;;
    5) test_step5 ;;
    6) test_step6 ;;
    all)
        passed=0
        failed=0
        for step in 1 2 3 4 5 6; do
            test_step$step
            if [ $? -eq 0 ]; then
                ((passed++))
            else
                ((failed++))
            fi
            echo ""
        done
        echo "==============================="
        echo -e "Results: ${GREEN}${passed} passed${NC}, ${RED}${failed} failed${NC}"
        echo "==============================="
        ;;
    *) echo "Unknown step: $1" ;;
esac
