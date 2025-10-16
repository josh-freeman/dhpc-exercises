import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np

# Input sizes to test
input_sizes = [(10**(x+3)) for x in range(7)]
configs = [
    ('exercise_2.c', '-O3', '-O3'),
    ('exercise_2_manual_opt.c', '-O3', 'manual_opt -O3'),
    ('exercise_2_manual_opt_padded.c', '-O3', 'manual_opt_padded -O3'),
    ('exercise_2_manual_opt_pragma_for.c', '-O3', 'manual_opt_pragma_for -O3')
]

# Store results for each configuration
results = {label: [] for _, _, label in configs}

# Compile and run for each configuration
for source_file, opt, label in configs:
    print(f"\nCompiling {source_file} with {opt}...")
    compile_result = subprocess.run(['gcc', opt, '-fopenmp', '-o', 'exercise_2', source_file],
                                   capture_output=True, text=True)
    if compile_result.returncode != 0:
        print(f"Compilation failed for {label}: {compile_result.stderr}")
        continue

    # Run for each input size
    for size in input_sizes:
        num_runs = 5
        tpe_measurements = []
        for run in range(num_runs):
            result = subprocess.run(['./exercise_2', str(size)],
                                  capture_output=True, text=True)
            output = json.loads(result.stdout.strip())
            run_time = output['run_time']
            tpe_measurements.append(run_time)

        mean_time = np.mean(tpe_measurements)
        std_time = np.std(tpe_measurements)
        results[label].append((mean_time, std_time))
        print(f"{label} N={size}: Time={mean_time:.6f} ± {std_time:.6f}s")

# Plot results
plt.figure(figsize=(12, 7))
for label in results:
    if results[label]:
        means = [r[0] for r in results[label]]
        stds = [r[1] for r in results[label]]
        plt.errorbar(input_sizes, means, yerr=stds, fmt='o-', linewidth=2,
                    markersize=8, capsize=5, capthick=2, label=label)

plt.xlabel('Input Size (num_steps)', fontsize=12)
plt.ylabel('Runtime (seconds)', fontsize=12)
plt.xscale('log', base=2)
plt.xticks(input_sizes, [f"{x:,}" for x in input_sizes])
plt.title('Runtime vs Input Size (Different Optimization Levels)', fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('time_vs_size.png', dpi=300)
plt.show()

print(f"\nPlot saved as 'time_vs_size.png'")
