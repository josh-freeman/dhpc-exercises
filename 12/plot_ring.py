#!/usr/bin/env python3
"""
Plotting script for MPI ring benchmark results.
Usage: python3 plot_ring.py

Expects data files: results_2.dat, results_4.dat, results_8.dat
Each file has columns: bytes time_us
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def read_data(filename):
    """Read benchmark data, skipping comment lines."""
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found, skipping")
        return None
    data = np.loadtxt(filename, comments='#')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data

# Load results for different process counts
data_files = {
    '2 procs': 'results_2.dat',
    '4 procs': 'results_4.dat',
    '8 procs': 'results_8.dat',
}

results = {}
for label, filename in data_files.items():
    data = read_data(filename)
    if data is not None:
        results[label] = data

if not results:
    print("No data files found! Run the benchmark first:")
    print("  mpirun -np 2 ./step4 > results_2.dat")
    print("  mpirun -np 4 ./step4 > results_4.dat")
    print("  mpirun -np 8 ./step4 > results_8.dat")
    sys.exit(1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

markers = ['o', 's', '^', 'D']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# Column indices: 0=bytes, 1=time_us

# Plot 1: Latency vs Message Size
for i, (label, data) in enumerate(results.items()):
    bytes_col = data[:, 0]
    time_col = data[:, 1]
    # Replace 0-byte with 0.5 for log scale display
    bytes_display = np.where(bytes_col == 0, 0.5, bytes_col)
    ax1.plot(bytes_display, time_col, f'{markers[i]}-',
             label=label, linewidth=2, markersize=8, color=colors[i])

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Message size (bytes)', fontsize=12)
ax1.set_ylabel('Round-trip time (us)', fontsize=12)
ax1.set_title('Ring Latency vs Message Size', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, which="both", ls="-", alpha=0.3)

# Plot 2: Bandwidth vs Message Size (skip 0-byte messages)
for i, (label, data) in enumerate(results.items()):
    mask = data[:, 0] > 0
    bytes_col = data[mask, 0]
    time_col = data[mask, 1]
    # Bandwidth = total bytes transferred / time
    # In a ring of P procs, each hop transfers 'bytes' once, P hops total
    # But the interesting metric is single-hop bandwidth: bytes / (time_us / P)
    # Or simply: bytes / time_us * 1e6 / (1024*1024) for MB/s throughput
    bandwidth = bytes_col / (time_col * 1e-6) / (1024 * 1024)
    ax2.semilogx(bytes_col, bandwidth, f'{markers[i]}-',
                 label=label, linewidth=2, markersize=8, color=colors[i])

ax2.set_xlabel('Message size (bytes)', fontsize=12)
ax2.set_ylabel('Throughput (MB/s)', fontsize=12)
ax2.set_title('Ring Throughput vs Message Size', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, which="both", ls="-", alpha=0.3)

plt.tight_layout()
plt.savefig('ring_benchmark.png', dpi=150)
print("Plot saved to ring_benchmark.png")

# Print summary
print("\n=== Summary ===")
for label, data in results.items():
    print(f"\n{label}:")
    for row in data:
        bstr = f"{int(row[0]):>8d} bytes" if row[0] > 0 else "   0 bytes"
        print(f"  {bstr}: {row[1]:>10.2f} us")
