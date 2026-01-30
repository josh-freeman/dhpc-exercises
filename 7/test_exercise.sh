#!/bin/bash
#
# Test script for Recitation 7: Cache Organization and Coherence
#
# Usage:
#   ./test_exercise.sh           # Run all tests
#   ./test_exercise.sh cache     # Run cache size detection
#   ./test_exercise.sh verify    # Verify against hwloc/lscpu
#

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "  Recitation 7: Cache Organization"
echo "========================================"
echo

run_cache_test() {
    echo -e "${YELLOW}Testing: Cache Size Detection${NC}"
    echo "----------------------------------------"

    # Compile
    echo "Compiling cache_size_detector.c..."
    gcc -O2 -o cache_size_detector cache_size_detector.c
    if [ $? -ne 0 ]; then
        echo -e "${RED}FAILED: Compilation error${NC}"
        return 1
    fi
    echo -e "${GREEN}Compilation successful${NC}"

    # Run (with reduced iterations for quick test)
    echo
    echo "Running cache size detector (this may take a minute)..."
    echo "Output will be saved to cache_results.csv"
    echo
    ./cache_size_detector > cache_results.csv

    if [ $? -ne 0 ]; then
        echo -e "${RED}FAILED: Runtime error${NC}"
        return 1
    fi

    # Check output
    if [ ! -f cache_results.csv ]; then
        echo -e "${RED}FAILED: No output file created${NC}"
        return 1
    fi

    lines=$(wc -l < cache_results.csv)
    if [ "$lines" -lt 10 ]; then
        echo -e "${RED}FAILED: Output file too small${NC}"
        return 1
    fi

    echo -e "${GREEN}PASSED: Cache detector completed with $lines data points${NC}"

    # Try to generate plot
    echo
    echo "Generating plot..."
    if command -v python3 &> /dev/null; then
        python3 plot_cache.py 2>/dev/null || echo "(Plot generation may require matplotlib)"
    else
        echo "(Python3 not found, skipping plot)"
    fi

    return 0
}

run_verify() {
    echo -e "${YELLOW}Verifying: System Cache Information${NC}"
    echo "----------------------------------------"

    echo "From lscpu:"
    lscpu | grep -i cache || echo "(lscpu cache info not available)"

    echo
    echo "From /sys/devices/system/cpu/cpu0/cache/:"
    if [ -d /sys/devices/system/cpu/cpu0/cache ]; then
        for i in /sys/devices/system/cpu/cpu0/cache/index*/; do
            if [ -f "${i}level" ] && [ -f "${i}size" ] && [ -f "${i}type" ]; then
                level=$(cat "${i}level")
                size=$(cat "${i}size")
                type=$(cat "${i}type")
                echo "  L${level} ${type}: ${size}"
            fi
        done
    else
        echo "(Cache info not available in sysfs)"
    fi

    echo
    if command -v hwloc-ls &> /dev/null; then
        echo "From hwloc:"
        hwloc-ls --only cache 2>/dev/null || echo "(hwloc error)"
    else
        echo "hwloc not installed. Install with: sudo apt install hwloc"
    fi

    echo
    echo -e "${GREEN}Verification complete${NC}"
    return 0
}

show_help() {
    echo "Usage: $0 [test]"
    echo
    echo "Tests:"
    echo "  cache   - Run cache size detection test"
    echo "  verify  - Show system cache information"
    echo "  all     - Run all tests (default)"
    echo
    echo "Example:"
    echo "  $0 cache    # Run cache detection"
    echo "  $0 verify   # Compare with hwloc/lscpu"
}

# Main
case "${1:-all}" in
    cache)
        run_cache_test
        ;;
    verify)
        run_verify
        ;;
    all)
        run_cache_test
        echo
        run_verify
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown test: $1${NC}"
        show_help
        exit 1
        ;;
esac

echo
echo "========================================"
echo "  Test Complete"
echo "========================================"
