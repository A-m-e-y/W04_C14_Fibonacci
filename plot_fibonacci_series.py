import pandas as pd
import matplotlib.pyplot as plt

# Load CSV with GPU total time
df = pd.read_csv("fibonacci_series_timing.csv")

# Extract columns
powers = df['PowerOf2']
labels = [f"2^{p}" for p in powers]
gpu_kernel = df['Kernel_ms']
gpu_total = df['GPU_Total_ms']
cpu_time = df['CPU_ms']

# Plot setup
plt.figure(figsize=(16, 6))

# GPU Kernel Time
plt.plot(labels, gpu_kernel, marker='o', linestyle='-', color='navy', label='GPU Kernel Time (Fibonacci)')

# GPU Total Time
plt.plot(labels, gpu_total, marker='^', linestyle='--', color='royalblue', label='GPU Total Time (Fibonacci)')

# CPU Time
plt.plot(labels, cpu_time, marker='s', linestyle='-', color='indianred', label='CPU Compute Time (Fibonacci)')

# Log scale
plt.yscale("log")
plt.grid(True, which="both", linestyle='--', linewidth=0.5)

# Labels and title
plt.xlabel("Matrix Size (N = 2^p)")
plt.ylabel("Execution Time (ms, log scale)")
plt.title("Fibonacci GPU vs CPU Execution Time Comparison")
plt.legend()
plt.tight_layout()

# Save and show
plt.savefig("plots/fibonacci_series_kernel_vs_cpu_total.png")
plt.show()
