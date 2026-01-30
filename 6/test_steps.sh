#!/bin/bash

# Test script for MPI RMA exercise
# Run: ./test_steps.sh [step_number]

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

test_step1() {
    echo -e "${YELLOW}Testing Step 1: RMA Window Creation${NC}"

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
    count=$(echo "$output" | grep -c "local value")

    if [ "$count" -eq 4 ]; then
        echo -e "${GREEN}PASS: All 4 ranks created windows and printed values${NC}"
        echo "Output:"
        echo "$output"
        return 0
    else
        echo -e "${RED}FAIL: Expected 4 outputs, got $count${NC}"
        echo "Output:"
        echo "$output"
        return 1
    fi
}

test_step2() {
    echo -e "${YELLOW}Testing Step 2: Ring Put with Fence${NC}"

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
    correct_count=$(echo "$output" | grep -c "CORRECT")

    if [ "$correct_count" -eq 4 ]; then
        echo -e "${GREEN}PASS: Ring Put working correctly${NC}"
        echo "Output:"
        echo "$output"
        return 0
    else
        echo -e "${RED}FAIL: Only $correct_count/4 ranks received correct data${NC}"
        echo "Output:"
        echo "$output"
        return 1
    fi
}

test_step3() {
    echo -e "${YELLOW}Testing Step 3: 1D Halo Exchange${NC}"

    if [ ! -f step3.c ]; then
        echo -e "${RED}FAIL: step3.c not found${NC}"
        return 1
    fi

    mpicc -o step3 step3.c 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}FAIL: Compilation failed${NC}"
        return 1
    fi

    output=$(mpirun -np 4 ./step3 2>/dev/null)
    correct_count=$(echo "$output" | grep -c "correct")
    error_count=$(echo "$output" | grep -c "wrong")

    if [ "$correct_count" -eq 4 ] && [ "$error_count" -eq 0 ]; then
        echo -e "${GREEN}PASS: 1D Halo exchange working correctly${NC}"
        echo "Output:"
        echo "$output"
        return 0
    else
        echo -e "${RED}FAIL: Halo exchange has errors${NC}"
        echo "Output:"
        echo "$output"
        return 1
    fi
}

test_step4() {
    echo -e "${YELLOW}Testing Step 4: 2D Heat Diffusion${NC}"

    if [ ! -f step4.c ]; then
        echo -e "${RED}FAIL: step4.c not found${NC}"
        return 1
    fi

    mpicc -O2 -o step4 step4.c -lm 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}FAIL: Compilation failed${NC}"
        return 1
    fi

    output=$(mpirun -np 4 ./step4 2>/dev/null)

    if echo "$output" | grep -q "Simulation complete"; then
        echo -e "${GREEN}PASS: Simulation completed${NC}"
        echo "Output:"
        echo "$output"

        # Check if output files were created
        if ls heat_output_*.dat 1>/dev/null 2>&1; then
            echo ""
            echo "Output files created. Run 'python3 plot_heat.py' to visualize."
        fi
        return 0
    else
        echo -e "${RED}FAIL: Simulation did not complete${NC}"
        echo "Output:"
        echo "$output"
        return 1
    fi
}

# Main
if [ -z "$1" ]; then
    echo "Usage: ./test_steps.sh [1|2|3|4|all]"
    echo ""
    echo "Tests your progress through the MPI RMA exercise steps:"
    echo "  1 - Window Creation"
    echo "  2 - Ring Put with Fence"
    echo "  3 - 1D Halo Exchange"
    echo "  4 - 2D Heat Diffusion"
    exit 0
fi

case $1 in
    1) test_step1 ;;
    2) test_step2 ;;
    3) test_step3 ;;
    4) test_step4 ;;
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
