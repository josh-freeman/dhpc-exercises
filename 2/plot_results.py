import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np

# Input sizes to test
input_sizes = [100, 200, 500, 1000, 1024, 1048, 1400, 1440, 1500, 6000, 10000]
configs = [
    ('../exericse_1/exercise_1_manual_opt.c', '-O3', 'exercise_1_manual_opt', 'exercise_1'),
    ('exercise_2_assignment.c', '-O3', 'exercise_2_assignment', 'exercise_2'),
]

# Store results for each configuration
results = {label: [] for _, _, label, _ in configs}

# Compile and run for each configuration
for source_file, opt, label, exe_name in configs:
    print(f"\nCompiling {source_file} with {opt}...")
    compile_result = subprocess.run(['gcc', opt, '-fopenmp', '-o', exe_name, source_file, '-lm'],
                                   capture_output=True, text=True)
    if compile_result.returncode != 0:
        print(f"Compilation failed for {label}: {compile_result.stderr}")
        continue

    # Run for each input size
    for size in input_sizes:
        num_runs = 5
        cycle_measurements = []
        for run in range(num_runs):
            result = subprocess.run([f'./{exe_name}', str(size)],
                                  capture_output=True, text=True)
            output = json.loads(result.stdout.strip())
            cycles = output['cycles']
            cycle_measurements.append(cycles)

        mean_cycles = np.mean(cycle_measurements)
        std_cycles = np.std(cycle_measurements)
        results[label].append((mean_cycles, std_cycles))
        print(f"{label} N={size}: Cycles={mean_cycles:.0f} ± {std_cycles:.0f}")

# Calculate and print speedup
print("\n" + "="*60)
print("SPEEDUP ANALYSIS: exercise_2_assignment vs exercise_1_manual_opt")
print("="*60)

baseline_label = 'exercise_1_manual_opt'
optimized_label = 'exercise_2_assignment'

if results[baseline_label] and results[optimized_label]:
    speedups = []
    print(f"\n{'N':>8} | {'Baseline Cycles':>16} | {'Optimized Cycles':>16} | {'Speedup':>10}")
    print("-" * 60)

    for i, size in enumerate(input_sizes):
        baseline_cycles = results[baseline_label][i][0]
        optimized_cycles = results[optimized_label][i][0]
        speedup = baseline_cycles / optimized_cycles
        speedups.append(speedup)
        print(f"{size:>8} | {baseline_cycles:>16.0f} | {optimized_cycles:>16.0f} | {speedup:>10.2f}x")

    avg_speedup = np.mean(speedups)
    min_speedup = np.min(speedups)
    max_speedup = np.max(speedups)

    print("-" * 60)
    print(f"\nSpeedup Summary:")
    print(f"  Average speedup: {avg_speedup:.2f}x")
    print(f"  Min speedup:     {min_speedup:.2f}x (N={input_sizes[np.argmin(speedups)]})")
    print(f"  Max speedup:     {max_speedup:.2f}x (N={input_sizes[np.argmax(speedups)]})")

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Cycles vs Input Size
for label in results:
    if results[label]:
        means = [r[0] for r in results[label]]
        stds = [r[1] for r in results[label]]
        ax1.errorbar(input_sizes, means, yerr=stds, fmt='o-', linewidth=2,
                    markersize=8, capsize=5, capthick=2, label=label)

ax1.set_xlabel('Input Size (N)', fontsize=12)
ax1.set_ylabel('Cycles', fontsize=12)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title('Cycles vs Input Size', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Speedup vs Input Size
if results[baseline_label] and results[optimized_label]:
    ax2.plot(input_sizes, speedups, 'go-', linewidth=2, markersize=10, label='Speedup')
    ax2.axhline(y=1.0, color='r', linestyle='--', label='No speedup (1x)')
    ax2.axhline(y=avg_speedup, color='b', linestyle=':', label=f'Avg speedup ({avg_speedup:.2f}x)')
    ax2.set_xlabel('Input Size (N)', fontsize=12)
    ax2.set_ylabel('Speedup (x)', fontsize=12)
    ax2.set_xscale('log')
    ax2.set_title(f'Speedup: exercise_2_assignment over exercise_1_manual_opt\n(Avg: {avg_speedup:.2f}x)', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('speedup_comparison.png', dpi=300)
plt.show()

print(f"\nPlot saved as 'speedup_comparison.png'")
