# W04_C14_Fibonacci
Course: HW for AI &amp; ML, Week 4 Challenge 14, Benchmarking Fibonacci sequence in CUDA.

# GPU (CUDA) vs CPU Performance for Fibonacci Sequence Computation

## TL;DR

Even though the **GPU kernel execution is significantly faster than the CPU**, the **total GPU execution time is more than 2x slower** due to massive overhead from **memory transfers and allocations**.

| Metric          | Time (ms)     |
| --------------- | ------------- |
| CPU Total Time  | **325.16 ms** |
| GPU Total Time  | **684.09 ms** |
| GPU Kernel Time | **215.75 ms** |

> **Conclusion:** The GPU spends less than 1/3 of its time doing useful computation. The rest is lost to `cudaMalloc`, `cudaMemcpy`, and `cudaFree`.

---

## 🔧 Experimental Setup

* **Target Computation**: Fibonacci sequence up to $N = 2^{24} = 16,777,216$
* **Platform**: CUDA running on NVIDIA `RTX-3050` 4GB GPU, WSL2 Linux subsystem and Intel `i5-12450H` CPU
* **Comparison**: Sequential CPU vs single-threaded CUDA GPU kernel
* **Metrics recorded**:

  * cudaMalloc Time
  * Kernel Time
  * D2H memcpy Time
  * cudaFree Time
  * Total GPU Time (sum of above)
  * CPU Time (sequential loop)

All times measured using `cudaEvent_t` for GPU and `clock_gettime()` for CPU.

---

## 📊 Full Metrics Breakdown

| Component        | Time (ms)  | Share of GPU Total (%) |
| ---------------- | ---------- | ---------------------- |
| cudaMalloc       | 59.33      | 8.67%                  |
| H2D Memcpy       | 0.00       | 0.00%                  |
| Kernel Execution | 215.75     | 31.53%                 |
| D2H Memcpy       | 368.09     | 53.80%                 |
| cudaFree         | 39.96      | 5.84%                  |
| **GPU Total**    | **684.09** | 100%                   |
| **CPU Total**    | **325.16** |                        |

---

## 🔬 Observations

### ✅ GPU Kernel is Fast

* Computing 16 million Fibonacci numbers in 215 ms is **very fast**, especially with a naïve loop.
* GPU excels at math-heavy tasks when memory isn't a bottleneck.

### ❌ Memory Copy and Setup Dominate

* `cudaMemcpy` from device to host takes **368 ms**, over **half** of total GPU time.
* `cudaMalloc` + `cudaFree` together take \~**100 ms**.
* So even though the kernel is fast, the total time becomes worse than CPU.

### ✅ CPU Loop is Efficient

* The CPU completes the task in \~325 ms.
* Despite being sequential, **data is already local**, no transfer overhead.
* No memory allocation/deallocation costs mid-loop.

---

## 📈 Visualization

<img src="plots/fibonacci_gpu_vs_cpu_bar.png" width="800">


* Top bar: **Stacked GPU timing** — shows how much of GPU time is wasted outside the kernel
* Bottom bar: **CPU total time** — a single, tight gray bar
* **Observation**: GPU bar is nearly 2x longer; only a fraction of it is actual compute.

---

## 🧪 Key Lessons

1. **Don’t judge GPU performance by kernel time alone.** Measure total pipeline time.
2. **Memory operations kill performance** when working with massive arrays.
3. For **serial or low-arithmetic tasks**, CPU might outperform GPU.
4. **cudaMemcpy and cudaMalloc** should be reused and minimized if possible.

---

## 🗂 Files

* `fibonacci_gpu_vs_cpu.cu`: Benchmark source code
* `fibonacci_timing.csv`: Output metrics
* `plot_fibonacci_single.py`: Script for generating the stacked bar plot
* `plots/fibonacci_gpu_vs_cpu_bar.png`: Visualization output
* `README.md`: Documentation
---
