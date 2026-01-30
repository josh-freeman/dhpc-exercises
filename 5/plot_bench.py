import matplotlib.pyplot as plt
import numpy as np

# Read data from files
def read_data(filename):
    data = np.loadtxt(filename)
    return data

# Load results for different process counts
results_2 = read_data('results_2.dat')
results_4 = read_data('results_4.dat')
results_8 = read_data('results_8.dat')

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Column indices: 0=count, 1=bytes_per_rank, 2=total_bytes, 3=time_us, 4=bandwidth

# Plot 1: Time vs Data Size
ax1.loglog(results_2[:, 1], results_2[:, 3], 'o-', label='2 processes', linewidth=2, markersize=8)
ax1.loglog(results_4[:, 1], results_4[:, 3], 's-', label='4 processes', linewidth=2, markersize=8)
ax1.loglog(results_8[:, 1], results_8[:, 3], '^-', label='8 processes', linewidth=2, markersize=8)

ax1.set_xlabel('Bytes per rank', fontsize=12)
ax1.set_ylabel('Time (microseconds)', fontsize=12)
ax1.set_title('MPI_Allgather Latency vs Data Size', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, which="both", ls="-", alpha=0.3)

# Plot 2: Bandwidth vs Data Size
ax2.semilogx(results_2[:, 1], results_2[:, 4], 'o-', label='2 processes', linewidth=2, markersize=8)
ax2.semilogx(results_4[:, 1], results_4[:, 4], 's-', label='4 processes', linewidth=2, markersize=8)
ax2.semilogx(results_8[:, 1], results_8[:, 4], '^-', label='8 processes', linewidth=2, markersize=8)

ax2.set_xlabel('Bytes per rank', fontsize=12)
ax2.set_ylabel('Bandwidth (MB/s)', fontsize=12)
ax2.set_title('MPI_Allgather Bandwidth vs Data Size', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, which="both", ls="-", alpha=0.3)

plt.tight_layout()
plt.savefig('allgather_benchmark.png', dpi=150)
plt.close()

print("Plot saved to allgather_benchmark.png")

# Print summary table
print("\n=== Summary ===")
print(f"{'Processes':<12} {'Small msg (1 double)':<25} {'Large msg (100k doubles)':<25}")
print(f"{'':12} {'Time(us)':<12} {'BW(MB/s)':<12} {'Time(us)':<12} {'BW(MB/s)':<12}")
print("-" * 70)
print(f"{'2':<12} {results_2[0,3]:<12.2f} {results_2[0,4]:<12.2f} {results_2[-1,3]:<12.2f} {results_2[-1,4]:<12.2f}")
print(f"{'4':<12} {results_4[0,3]:<12.2f} {results_4[0,4]:<12.2f} {results_4[-1,3]:<12.2f} {results_4[-1,4]:<12.2f}")
print(f"{'8':<12} {results_8[0,3]:<12.2f} {results_8[0,4]:<12.2f} {results_8[-1,3]:<12.2f} {results_8[-1,4]:<12.2f}")
