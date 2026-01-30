#!/usr/bin/env python3
"""
Plotting script for MPI_Allgather benchmark results.
Usage: python3 plot_results.py

Expects data files: results_2.dat, results_4.dat, results_8.dat
Each file should have columns: count bytes_per_rank total_bytes time_us bandwidth
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
    print("  mpirun -np 2 ./benchmark > results_2.dat")
    print("  mpirun -np 4 ./benchmark > results_4.dat")
    print("  mpirun -np 8 ./benchmark > results_8.dat")
    sys.exit(1)

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

markers = ['o', 's', '^', 'D']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# Column indices: 0=count, 1=bytes_per_rank, 2=total_bytes, 3=time_us, 4=bandwidth

# Plot 1: Time vs Data Size
for i, (label, data) in enumerate(results.items()):
    ax1.loglog(data[:,1], data[:,3], f'{markers[i]}-',
               label=label, linewidth=2, markersize=8, color=colors[i])

ax1.set_xlabel('Bytes per rank', fontsize=12)
ax1.set_ylabel('Time (microseconds)', fontsize=12)
ax1.set_title('MPI_Allgather Latency vs Data Size', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, which="both", ls="-", alpha=0.3)

# Plot 2: Bandwidth vs Data Size
for i, (label, data) in enumerate(results.items()):
    ax2.semilogx(data[:,1], data[:,4], f'{markers[i]}-',
                 label=label, linewidth=2, markersize=8, color=colors[i])

ax2.set_xlabel('Bytes per rank', fontsize=12)
ax2.set_ylabel('Bandwidth (MB/s)', fontsize=12)
ax2.set_title('MPI_Allgather Bandwidth vs Data Size', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, which="both", ls="-", alpha=0.3)

plt.tight_layout()
plt.savefig('benchmark_results.png', dpi=150)
print("Plot saved to benchmark_results.png")

# Print summary
print("\n=== Summary ===")
for label, data in results.items():
    print(f"\n{label}:")
    print(f"  Smallest msg: {data[0,3]:.2f} us, {data[0,4]:.2f} MB/s")
    print(f"  Largest msg:  {data[-1,3]:.2f} us, {data[-1,4]:.2f} MB/s")
