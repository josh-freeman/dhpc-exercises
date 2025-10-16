import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np

# Input sizes to test
input_sizes = [20*(2**x) for x in range(6)]
configs = [
    ('exercise_1.c', '-O3', '-O3'),
    ('exercise_1_manual_opt.c', '-O3', 'manual_opt -O3')
]

# Store results for each configuration
results = {label: [] for _, _, label in configs}

# Compile and run for each configuration
for source_file, opt, label in configs:
    print(f"\nCompiling {source_file} with {opt}...")
    compile_result = subprocess.run(['gcc', opt, '-o', 'exercise_1', source_file],
                                   capture_output=True, text=True)
    if compile_result.returncode != 0:
        print(f"Compilation failed for {label}: {compile_result.stderr}")
        continue

    # Run for each input size
    for size in input_sizes:
        num_runs = 5
        tpe_measurements = []
        for run in range(num_runs):
            result = subprocess.run(['./exercise_1', str(size)],
                                  capture_output=True, text=True)
            output = json.loads(result.stdout.strip())
            cycles = output['cycles']
            num_steps = 100
            num_elements = size * size
            tpe = cycles / (num_elements * num_steps)
            tpe_measurements.append(tpe)

        mean_tpe = np.mean(tpe_measurements)
        std_tpe = np.std(tpe_measurements)
        results[label].append((mean_tpe, std_tpe))
        print(f"{label} N={size}: TPE={mean_tpe:.4f} ± {std_tpe:.4f}")

# Plot results
plt.figure(figsize=(12, 7))
for label in results:
    if results[label]:
        means = [r[0] for r in results[label]]
        stds = [r[1] for r in results[label]]
        plt.errorbar(input_sizes, means, yerr=stds, fmt='o-', linewidth=2,
                    markersize=8, capsize=5, capthick=2, label=label)

plt.xlabel('Input Size (N)', fontsize=12)
plt.ylabel('Time per Element (Cycles/Element)', fontsize=12)
plt.xscale('log', base=2)
plt.xticks(input_sizes, [f"{x:,}" for x in input_sizes])
plt.title('Time per Element vs Input Size (Different Optimization Levels)', fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('time_vs_size.png', dpi=300)
plt.show()

print(f"\nPlot saved as 'time_vs_size.png'")
