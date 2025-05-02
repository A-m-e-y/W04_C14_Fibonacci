#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

__global__ void fibonacci_gpu(unsigned long long *output, int N)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        output[0] = 0;
        if (N > 1)
            output[1] = 1;
        for (int i = 2; i < N; i++)
        {
            output[i] = output[i - 1] + output[i - 2];
        }
    }
}

void fibonacci_cpu(unsigned long long *output, int N)
{
    output[0] = 0;
    if (N > 1)
        output[1] = 1;
    for (int i = 2; i < N; i++)
    {
        output[i] = output[i - 1] + output[i - 2];
    }
}

double get_cpu_time_ms()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char *argv[])
{
    int min_power = 15;
    int max_power = 25;

    if (argc >= 2)
    {
        max_power = atoi(argv[1]);
        if (max_power < min_power || max_power > 30)
        {
            printf("Invalid power value (must be between %d and 30)\n", min_power);
            return -1;
        }
    }

    FILE *fp = fopen("fibonacci_series_timing.csv", "w");
    if (!fp)
    {
        perror("Failed to open fibonacci_series_timing.csv");
        return -1;
    }
    fprintf(fp, "PowerOf2,N,Kernel_ms,CPU_ms,GPU_Total_ms\n");

    for (int power = min_power; power <= max_power; power++)
    {
        int N = 1 << power;

        unsigned long long *cpu_output = (unsigned long long *)malloc(N * sizeof(unsigned long long));
        unsigned long long *gpu_output = (unsigned long long *)malloc(N * sizeof(unsigned long long));
        unsigned long long *d_output;

        // --- CPU TIMING ---
        double cpu_start = get_cpu_time_ms();
        fibonacci_cpu(cpu_output, N);
        double cpu_end = get_cpu_time_ms();
        double cpu_time = cpu_end - cpu_start;

        // --- GPU TIMING ---
        cudaMalloc(&d_output, N * sizeof(unsigned long long));

        float kernel_time = 0.0f;
        float total_time = 0.0f;

        cudaEvent_t start, stop, total_start, total_stop;
        cudaEventCreate(&start); // for kernel
        cudaEventCreate(&stop);
        cudaEventCreate(&total_start); // for full GPU measurement
        cudaEventCreate(&total_stop);

        cudaEventRecord(total_start); // start total

        cudaEventRecord(start);
        fibonacci_gpu<<<1, 1>>>(d_output, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&kernel_time, start, stop);

        cudaMemcpy(gpu_output, d_output, N * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaFree(d_output);

        cudaEventRecord(total_stop); // stop total
        cudaEventSynchronize(total_stop);
        cudaEventElapsedTime(&total_time, total_start, total_stop);

        // --- Output ---
        printf("N = 2^%d (%d) | GPU Kernel = %.4f ms | GPU Total = %.4f ms | CPU = %.4f ms\n",
               power, N, kernel_time, total_time, cpu_time);

        fprintf(fp, "%d,%d,%.4f,%.4f,%.4f\n", power, N, kernel_time, cpu_time, total_time);

        free(cpu_output);
        free(gpu_output);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaEventDestroy(total_start);
        cudaEventDestroy(total_stop);
    }

    fclose(fp);
    return 0;
}
