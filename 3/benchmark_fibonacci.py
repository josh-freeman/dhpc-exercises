import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np

# Fibonacci sizes to test
fib_sizes = [2**i for i in range(12)]

configs = [
    ('fibonacci_seq.c', 'Sequential'),
    ('fibonacci_par.c', 'Parallel (Tasks)')
]

# Store results for each configuration
results = {label: [] for _, label in configs}

# Compile and run for each configuration
for source_file, label in configs:
    print(f"\nCompiling {source_file}...")
    compile_result = subprocess.run(['gcc', '-O3', '-fopenmp', '-o', source_file.replace('.c', ''), source_file],
                                   capture_output=True, text=True)
    if compile_result.returncode != 0:
        print(f"Compilation failed for {label}: {compile_result.stderr}")
        continue

    executable = './' + source_file.replace('.c', '')

    # Run for each Fibonacci size
    for N in fib_sizes:
        num_runs = 10
        time_measurements = []
        for run in range(num_runs):
            result = subprocess.run([executable, str(N)],
                                  capture_output=True, text=True)
            try:
                output = json.loads(result.stdout.strip())
                run_time = output['run_time']
                time_measurements.append(run_time)
            except json.JSONDecodeError:
                print(f"Failed to parse output for {label} N={N}: {result.stdout}")
                continue

        if time_measurements:
            mean_time = np.mean(time_measurements)
            std_time = np.std(time_measurements)
            results[label].append((mean_time, std_time))
            print(f"{label} N={N}: Time={mean_time:.9f} ± {std_time:.9f}s")
        else:
            results[label].append((0, 0))

# Plot results
plt.figure(figsize=(12, 7))
for label in results:
    if results[label]:
        means = [r[0] for r in results[label]]
        stds = [r[1] for r in results[label]]
        plt.errorbar(fib_sizes, means, yerr=stds, fmt='o-', linewidth=2,
                    markersize=8, capsize=5, capthick=2, label=label)

plt.xlabel('Fibonacci Number (N)', fontsize=12)
plt.ylabel('Runtime (seconds)', fontsize=12)
plt.title('Fibonacci Computation: Sequential vs Parallel (OpenMP Tasks)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fibonacci_comparison.png', dpi=300)
plt.show()

print(f"\nPlot saved as 'fibonacci_comparison.png'")

# Calculate speedup
if results['Sequential'] and results['Parallel (Tasks)']:
    print("\nSpeedup (Sequential / Parallel):")
    for i, N in enumerate(fib_sizes):
        seq_time = results['Sequential'][i][0]
        par_time = results['Parallel (Tasks)'][i][0]
        if par_time > 0:
            speedup = seq_time / par_time
            print(f"N={N}: {speedup:.2f}x")
