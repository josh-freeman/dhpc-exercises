#!/usr/bin/env python3
"""
Plot cache size detection results.

Usage:
    ./cache_size_detector > cache_results.csv
    python3 plot_cache.py

This script reads the CSV output from cache_size_detector and creates
a log-log plot showing memory access latency vs working set size.
Cache boundaries appear as step increases in latency.
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def load_data(filename='cache_results.csv'):
    """Load cache measurement data from CSV file."""
    sizes_kb = []
    latencies = []

    with open(filename, 'r') as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                sizes_kb.append(float(parts[0]))
                latencies.append(float(parts[3]))

    return np.array(sizes_kb), np.array(latencies)


def detect_cache_boundaries(sizes_kb, latencies, threshold=1.3):
    """
    Detect cache boundaries by finding significant jumps in latency.
    Returns list of (size_kb, latency_before, latency_after) tuples.
    """
    boundaries = []

    for i in range(1, len(latencies)):
        ratio = latencies[i] / latencies[i-1]
        if ratio > threshold:
            boundaries.append({
                'size_kb': sizes_kb[i],
                'size_before_kb': sizes_kb[i-1],
                'latency_before': latencies[i-1],
                'latency_after': latencies[i],
                'ratio': ratio
            })

    return boundaries


def format_size(size_kb):
    """Format size in human-readable form."""
    if size_kb >= 1024:
        return f"{size_kb/1024:.0f} MB"
    else:
        return f"{size_kb:.0f} KB"


def main():
    # Check if data file exists
    data_file = 'cache_results.csv'
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found.")
        print("Run './cache_size_detector > cache_results.csv' first.")
        sys.exit(1)

    # Load data
    print(f"Loading data from {data_file}...")
    sizes_kb, latencies = load_data(data_file)

    if len(sizes_kb) == 0:
        print("Error: No data found in file.")
        sys.exit(1)

    print(f"Loaded {len(sizes_kb)} data points")
    print(f"Size range: {format_size(min(sizes_kb))} to {format_size(max(sizes_kb))}")
    print(f"Latency range: {min(latencies):.1f} ns to {max(latencies):.1f} ns")

    # Detect cache boundaries
    boundaries = detect_cache_boundaries(sizes_kb, latencies)

    print("\n" + "="*60)
    print("Detected Cache Boundaries:")
    print("="*60)
    for i, b in enumerate(boundaries):
        print(f"  Cache L{i+1} size: ~{format_size(b['size_before_kb'])}")
        print(f"    Latency jump: {b['latency_before']:.1f} ns -> {b['latency_after']:.1f} ns ({b['ratio']:.2f}x)")
    print("="*60 + "\n")

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot data
    ax.loglog(sizes_kb, latencies, 'b.-', linewidth=2, markersize=8, label='Measured latency')

    # Add cache boundary annotations
    colors = ['green', 'orange', 'red', 'purple']
    cache_names = ['L1', 'L2', 'L3', 'L4']

    for i, b in enumerate(boundaries[:4]):  # At most 4 cache levels
        color = colors[i % len(colors)]
        name = cache_names[i]

        # Draw vertical line at boundary
        ax.axvline(x=b['size_before_kb'], color=color, linestyle='--', alpha=0.7, linewidth=2)

        # Add annotation
        y_pos = b['latency_after'] * 0.8
        ax.annotate(f'{name}\n~{format_size(b["size_before_kb"])}',
                   xy=(b['size_before_kb'], b['latency_before']),
                   xytext=(b['size_before_kb'] * 0.5, y_pos),
                   fontsize=11, fontweight='bold', color=color,
                   ha='center',
                   arrowprops=dict(arrowstyle='->', color=color, alpha=0.7))

    # Typical cache sizes (for reference lines)
    typical_caches = [
        (32, 'L1 (32KB typical)'),
        (256, 'L2 (256KB typical)'),
        (8192, 'L3 (8MB typical)'),
    ]

    for size, label in typical_caches:
        if min(sizes_kb) <= size <= max(sizes_kb):
            ax.axvline(x=size, color='gray', linestyle=':', alpha=0.3)

    # Labels and formatting
    ax.set_xlabel('Working Set Size', fontsize=12)
    ax.set_ylabel('Access Latency (ns)', fontsize=12)
    ax.set_title('Cache Size Detection via Memory Access Latency\n(Pointer Chasing Method)', fontsize=14)

    # Custom x-axis labels
    xticks = [1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    xticks = [x for x in xticks if min(sizes_kb) <= x <= max(sizes_kb)]
    xticklabels = [format_size(x) for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha='right')

    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)

    # Add text box with summary
    textstr = 'Cache boundaries detected:\n'
    for i, b in enumerate(boundaries[:3]):
        textstr += f'{cache_names[i]}: ~{format_size(b["size_before_kb"])}\n'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.02, textstr.strip(), transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()

    # Save plot
    output_file = 'cache_latency.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    # Also show plot if running interactively
    try:
        plt.show()
    except:
        pass

    # Print verification instructions
    print("\nTo verify these results, run:")
    print("  lscpu | grep -i cache")
    print("  hwloc-ls --only cache  # if hwloc is installed")


if __name__ == '__main__':
    main()
