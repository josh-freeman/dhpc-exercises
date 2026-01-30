#!/bin/bash

# Test script for MPI ring benchmark exercise
# Run: ./test_steps.sh [step_number]

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

test_step1() {
    echo -e "${YELLOW}Testing Step 1: Ring Communication${NC}"

    if [ ! -f step1.c ]; then
        echo -e "${RED}FAIL: step1.c not found${NC}"
        return 1
    fi

    mpicc -o step1 step1.c 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}FAIL: Compilation failed${NC}"
        mpicc -o step1 step1.c
        return 1
    fi

    output=$(mpirun --oversubscribe -np 4 ./step1 2>/dev/null)

    # Check rank 0 reports success
    if echo "$output" | grep -q "ring complete.*OK"; then
        echo -e "${GREEN}PASS: Ring completed with correct data${NC}"
    else
        echo -e "${RED}FAIL: Ring did not complete correctly${NC}"
        echo "Output was:"
        echo "$output"
        return 1
    fi

    # Check other ranks forwarded
    fwd_count=$(echo "$output" | grep -c "forwarded")
    if [ "$fwd_count" -eq 3 ]; then
        echo -e "${GREEN}PASS: All 3 intermediate ranks forwarded${NC}"
    else
        echo -e "${RED}FAIL: Expected 3 forwarding messages, got $fwd_count${NC}"
        return 1
    fi

    return 0
}

test_step2() {
    echo -e "${YELLOW}Testing Step 2: Ring with Timing${NC}"

    if [ ! -f step2.c ]; then
        echo -e "${RED}FAIL: step2.c not found${NC}"
        return 1
    fi

    mpicc -O2 -o step2 step2.c 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}FAIL: Compilation failed${NC}"
        mpicc -O2 -o step2 step2.c
        return 1
    fi

    output=$(mpirun --oversubscribe -np 4 ./step2 2>/dev/null)

    if echo "$output" | grep -qE "[0-9]+\.?[0-9]* *(us|microseconds|seconds)"; then
        echo -e "${GREEN}PASS: Timing output detected${NC}"
        echo "Output:"
        echo "$output"
    else
        echo -e "${RED}FAIL: No timing output found${NC}"
        echo "Output was:"
        echo "$output"
        return 1
    fi

    return 0
}

test_step3() {
    echo -e "${YELLOW}Testing Step 3: Robust Benchmarking${NC}"

    if [ ! -f step3.c ]; then
        echo -e "${RED}FAIL: step3.c not found${NC}"
        return 1
    fi

    mpicc -O2 -o step3 step3.c 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}FAIL: Compilation failed${NC}"
        mpicc -O2 -o step3 step3.c
        return 1
    fi

    output=$(mpirun --oversubscribe -np 4 ./step3 2>/dev/null)

    if echo "$output" | grep -qE "[0-9]+\.?[0-9]* *(us|microseconds|seconds)"; then
        echo -e "${GREEN}PASS: Average timing output detected${NC}"
        echo "Output:"
        echo "$output"
    else
        echo -e "${RED}FAIL: No timing output found (did you set WARMUP and ITERATIONS > 0?)${NC}"
        echo "Output was:"
        echo "$output"
        return 1
    fi

    # Run twice and compare for stability
    t1=$(mpirun --oversubscribe -np 4 ./step3 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1)
    t2=$(mpirun --oversubscribe -np 4 ./step3 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1)

    if [ -n "$t1" ] && [ -n "$t2" ]; then
        echo "  Run 1: ${t1}"
        echo "  Run 2: ${t2}"
        echo -e "${GREEN}(Check that these are reasonably close)${NC}"
    fi

    return 0
}

test_step4() {
    echo -e "${YELLOW}Testing Step 4: Message Size Sweep${NC}"

    if [ ! -f step4.c ]; then
        echo -e "${RED}FAIL: step4.c not found${NC}"
        return 1
    fi

    mpicc -O2 -o step4 step4.c 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}FAIL: Compilation failed${NC}"
        mpicc -O2 -o step4 step4.c
        return 1
    fi

    output=$(mpirun --oversubscribe -np 4 ./step4 2>/dev/null)
    # Count non-comment, non-empty data lines
    data_lines=$(echo "$output" | grep -v '^#' | grep -cE '^[0-9]')

    if [ "$data_lines" -ge 5 ]; then
        echo -e "${GREEN}PASS: Got $data_lines data lines${NC}"
        echo "Output:"
        echo "$output"
    else
        echo -e "${RED}FAIL: Expected at least 5 data lines, got $data_lines${NC}"
        echo "Output was:"
        echo "$output"
        return 1
    fi

    # Check that larger messages take more time (sanity)
    first_time=$(echo "$output" | grep -v '^#' | grep -E '^[0-9]' | head -1 | awk '{print $2}')
    last_time=$(echo "$output" | grep -v '^#' | grep -E '^[0-9]' | tail -1 | awk '{print $2}')

    if [ -n "$first_time" ] && [ -n "$last_time" ]; then
        echo "  Smallest msg time: ${first_time} us"
        echo "  Largest  msg time: ${last_time} us"
    fi

    return 0
}

test_step4_generate() {
    echo -e "${YELLOW}Step 4: Generating result files for plotting${NC}"

    if [ ! -f step4 ]; then
        echo -e "${RED}step4 binary not found. Run test_steps.sh 4 first.${NC}"
        return 1
    fi

    for np in 2 4 8; do
        echo "  Running with $np processes..."
        mpirun --oversubscribe -np $np ./step4 > results_${np}.dat 2>/dev/null
        if [ $? -eq 0 ]; then
            echo -e "  ${GREEN}results_${np}.dat created${NC}"
        else
            echo -e "  ${RED}Failed with $np processes${NC}"
        fi
    done

    return 0
}

# Main
if [ -z "$1" ]; then
    echo "Usage: ./test_steps.sh [1|2|3|4|generate|all]"
    echo ""
    echo "  1        - Test step 1 (ring communication)"
    echo "  2        - Test step 2 (timing)"
    echo "  3        - Test step 3 (robust benchmarking)"
    echo "  4        - Test step 4 (message size sweep)"
    echo "  generate - Run step4 with 2,4,8 procs to create result files"
    echo "  all      - Run all tests"
    exit 0
fi

case $1 in
    1) test_step1 ;;
    2) test_step2 ;;
    3) test_step3 ;;
    4) test_step4 ;;
    generate) test_step4_generate ;;
    all)
        test_step1
        echo ""
        test_step2
        echo ""
        test_step3
        echo ""
        test_step4
        ;;
    *) echo "Unknown step: $1" ;;
esac
