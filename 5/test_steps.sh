#!/bin/bash

# Test script for MPI benchmark exercise
# Run: ./test_steps.sh [step_number]

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

test_step1() {
    echo -e "${YELLOW}Testing Step 1: MPI Hello World${NC}"

    if [ ! -f step1.c ]; then
        echo -e "${RED}FAIL: step1.c not found${NC}"
        return 1
    fi

    mpicc -o step1 step1.c 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}FAIL: Compilation failed${NC}"
        return 1
    fi

    output=$(mpirun -np 4 ./step1 2>/dev/null)
    count=$(echo "$output" | grep -c "Hello from rank")

    if [ "$count" -eq 4 ]; then
        echo -e "${GREEN}PASS: All 4 ranks printed${NC}"
        return 0
    else
        echo -e "${RED}FAIL: Expected 4 lines, got $count${NC}"
        return 1
    fi
}

test_step2() {
    echo -e "${YELLOW}Testing Step 2: MPI_Allgather${NC}"

    if [ ! -f step2.c ]; then
        echo -e "${RED}FAIL: step2.c not found${NC}"
        return 1
    fi

    mpicc -o step2 step2.c 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}FAIL: Compilation failed${NC}"
        return 1
    fi

    output=$(mpirun -np 4 ./step2 2>/dev/null)

    # Check if all ranks received "10 20 30 40"
    if echo "$output" | grep -q "10 20 30 40"; then
        echo -e "${GREEN}PASS: Allgather working correctly${NC}"
        return 0
    else
        echo -e "${RED}FAIL: Data not gathered correctly${NC}"
        echo "Output was:"
        echo "$output"
        return 1
    fi
}

test_step3() {
    echo -e "${YELLOW}Testing Step 3: Basic Timing${NC}"

    if [ ! -f step3.c ]; then
        echo -e "${RED}FAIL: step3.c not found${NC}"
        return 1
    fi

    mpicc -O2 -o step3 step3.c 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}FAIL: Compilation failed${NC}"
        return 1
    fi

    output=$(mpirun -np 4 ./step3 2>/dev/null)
    count=$(echo "$output" | grep -c "seconds")

    if [ "$count" -ge 1 ]; then
        echo -e "${GREEN}PASS: Timing output detected${NC}"
        echo "Output:"
        echo "$output"
        return 0
    else
        echo -e "${RED}FAIL: No timing output${NC}"
        return 1
    fi
}

# Main
if [ -z "$1" ]; then
    echo "Usage: ./test_steps.sh [1|2|3|all]"
    echo ""
    echo "Tests your progress through the exercise steps."
    exit 0
fi

case $1 in
    1) test_step1 ;;
    2) test_step2 ;;
    3) test_step3 ;;
    all)
        test_step1
        echo ""
        test_step2
        echo ""
        test_step3
        ;;
    *) echo "Unknown step: $1" ;;
esac
