import pandas as pd
import math
import matplotlib.pyplot as plt

# Load the CSV and get the last row
df = pd.read_csv("fibonacci_timing.csv")
row = df.iloc[-1]  # last row

# Extract values
# power = row['PowerOf2']
N = row['N']
power = math.log2(N)
malloc_ms = row['Malloc_ms']
h2d_ms = row['H2D_ms']
kernel_ms = row['Kernel_ms']
d2h_ms = row['D2H_ms']
free_ms = row['Free_ms']
total_gpu_ms = row['TotalGPU_ms']
cpu_ms = row['CPU_ms']

# Stacked GPU components
gpu_components = [malloc_ms, h2d_ms, kernel_ms, d2h_ms, free_ms]
gpu_labels = ["cudaMalloc", "H2D Memcpy", "Kernel", "D2H Memcpy", "cudaFree"]
gpu_colors = ["#9c27b0", "#03a9f4", "#4caf50", "#fbc02d", "#ef5350"]

# Plot
plt.figure(figsize=(10, 4))
left = 0
for val, label, color in zip(gpu_components, gpu_labels, gpu_colors):
    plt.barh(y=0, width=val, left=left, label=label, color=color)
    plt.text(left + val / 2, 0, f"{val:.1f} ms", va='center', ha='center', fontsize=8, color='black')
    left += val

# CPU bar
plt.barh(y=1, width=cpu_ms, color='gray', label='CPU Total')
plt.text(cpu_ms / 2, 1, f"{cpu_ms:.1f} ms", va='center', ha='center', fontsize=9, color='white')

# Labels and formatting
plt.yticks([0, 1], ['GPU (stacked)', 'CPU (total)'])
plt.xlabel("Execution Time (ms)")
plt.title(f"Fibonacci Time Breakdown for N = {int(N)} (2^{int(power)})")
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("plots/fibonacci_gpu_vs_cpu_bar.png")
plt.show()
