#!/bin/bash
# Test script for SPIN/Promela exercises (Recitation 10)
# Usage: ./test_steps.sh <step_number>
#   step 1: simulation only (should print Hello messages)
#   step 2: verification (should find assertion violation)
#   step 3: verification (should find NO errors)
#   step 4: simulation (should show alternating MSG/ACK)
#   step 5: verification (should find assertion violation = solution)

set -e

STEP=$1
CLEANUP_FILES="pan pan.c pan.h pan.t pan.m pan.p pan.b pan.trail _spin_nvr.tmp"

cleanup() {
    rm -f $CLEANUP_FILES *.trail
}

trap cleanup EXIT

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}PASSED${NC} --- $1"; }
fail() { echo -e "${RED}FAILED${NC} --- $1"; exit 1; }
info() { echo -e "${YELLOW}INFO${NC} --- $1"; }

if [ -z "$STEP" ]; then
    echo "Usage: $0 <step_number (1-5)>"
    exit 1
fi

# Check that spin is installed
if ! command -v spin &> /dev/null; then
    echo "Error: 'spin' is not installed or not in PATH."
    echo "Install SPIN: https://spinroot.com/spin/whatispin.html"
    exit 1
fi

case $STEP in
    1)
        info "Step 1: Hello World (simulation)"
        FILE="step1_hello.pml"
        [ -f "$FILE" ] || fail "$FILE not found"

        OUTPUT=$(spin "$FILE" 2>&1)

        # Check for basic process output
        if echo "$OUTPUT" | grep -q "pid"; then
            pass "Processes created and printed PIDs."
        else
            fail "Expected output containing 'pid'. Got:\n$OUTPUT"
        fi
        echo "$OUTPUT"
        ;;

    2)
        info "Step 2: Broken Mutex (verification --- expect assertion violation)"
        FILE="step2_broken_mutex.pml"
        [ -f "$FILE" ] || fail "$FILE not found"

        spin -a "$FILE"
        gcc -O3 -o pan pan.c

        OUTPUT=$(./pan 2>&1 || true)
        echo "$OUTPUT"

        if echo "$OUTPUT" | grep -qi "assertion violated\|errors: 1"; then
            pass "SPIN found the assertion violation (broken mutex detected)."
        else
            fail "Expected SPIN to find an assertion violation."
        fi

        info "Replaying counterexample:"
        spin -t -p "$FILE" 2>&1 || true
        ;;

    3)
        info "Step 3: Peterson's Algorithm (verification --- expect NO errors)"
        FILE="step3_fixed_mutex.pml"
        [ -f "$FILE" ] || fail "$FILE not found"

        spin -a "$FILE"
        gcc -O3 -o pan pan.c

        OUTPUT=$(./pan 2>&1 || true)
        echo "$OUTPUT"

        if echo "$OUTPUT" | grep -qi "errors: 0"; then
            pass "SPIN verified Peterson's algorithm: no errors found."
        elif echo "$OUTPUT" | grep -qi "assertion violated\|errors: [1-9]"; then
            fail "SPIN found errors --- Peterson's implementation is incorrect."
        else
            fail "Unexpected verification output."
        fi
        ;;

    4)
        info "Step 4: Channel Communication (simulation)"
        FILE="step4_channels.pml"
        [ -f "$FILE" ] || fail "$FILE not found"

        OUTPUT=$(spin "$FILE" 2>&1)
        echo "$OUTPUT"

        # Check that messages were exchanged
        if echo "$OUTPUT" | grep -qi "MSG\|ACK\|sent\|got"; then
            pass "Channel communication observed."
        else
            fail "Expected MSG/ACK output from sender/receiver."
        fi

        # Also run verification
        info "Running verification..."
        spin -a "$FILE"
        gcc -O3 -o pan pan.c
        VOUTPUT=$(./pan 2>&1 || true)
        echo "$VOUTPUT"

        if echo "$VOUTPUT" | grep -qi "errors: 0"; then
            pass "Verification passed: no errors."
        else
            info "Verification found issues (check assertions)."
        fi
        ;;

    5)
        info "Step 5: Bridge Crossing Riddle (verification --- expect counterexample)"
        FILE="step5_bridge.pml"
        [ -f "$FILE" ] || fail "$FILE not found"

        spin -a "$FILE"
        gcc -O3 -o pan pan.c

        OUTPUT=$(./pan 2>&1 || true)
        echo "$OUTPUT"

        if echo "$OUTPUT" | grep -qi "assertion violated\|errors: 1"; then
            pass "SPIN found a solution (counterexample to impossibility assertion)."
            echo ""
            info "Replaying solution:"
            spin -t -p "$FILE" 2>&1 || true
        else
            fail "SPIN could not find a solution. Check your model."
        fi
        ;;

    *)
        echo "Unknown step: $STEP (valid: 1-5)"
        exit 1
        ;;
esac
