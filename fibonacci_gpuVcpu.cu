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
    int power = 20;
    if (argc >= 2)
    {
        power = atoi(argv[1]);
        if (power < 1)
        {
            printf("Invalid power (must be at least greater than 1)\n");
            return -1;
        }
    }
    int N = 1 << power;

    FILE *fp = fopen("fibonacci_timing.csv", "w");
    if (!fp)
    {
        perror("Failed to open CSV");
        return -1;
    }
    fprintf(fp, "N,Malloc_ms,H2D_ms,Kernel_ms,D2H_ms,Free_ms,TotalGPU_ms,CPU_ms\n");

    // CPU Fibonacci
    unsigned long long *cpu_output = (unsigned long long *)malloc(N * sizeof(unsigned long long));
    double cpu_start = get_cpu_time_ms();
    fibonacci_cpu(cpu_output, N);
    double cpu_end = get_cpu_time_ms();
    double cpu_time = cpu_end - cpu_start;

    // GPU allocations
    unsigned long long *d_output, *h_output;
    h_output = (unsigned long long *)malloc(N * sizeof(unsigned long long));

    float malloc_time = 0.0f, h2d_time = 0.0f, kernel_time = 0.0f;
    float d2h_time = 0.0f, free_time = 0.0f, total_time = 0.0f;

    cudaEvent_t malloc_start, malloc_stop, kernel_start, kernel_stop;
    cudaEvent_t d2h_start, d2h_stop, free_start, free_stop;
    cudaEvent_t total_start, total_stop;

    cudaEventCreate(&malloc_start);
    cudaEventCreate(&malloc_stop);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    cudaEventCreate(&d2h_start);
    cudaEventCreate(&d2h_stop);
    cudaEventCreate(&free_start);
    cudaEventCreate(&free_stop);
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);

    cudaEventRecord(total_start);

    // Malloc
    cudaEventRecord(malloc_start);
    cudaMalloc(&d_output, N * sizeof(unsigned long long));
    cudaEventRecord(malloc_stop);
    cudaEventSynchronize(malloc_stop);
    cudaEventElapsedTime(&malloc_time, malloc_start, malloc_stop);

    // H2D (not needed, set to 0)
    h2d_time = 0.0f;

    // Kernel
    cudaEventRecord(kernel_start);
    fibonacci_gpu<<<1, 1>>>(d_output, N);
    cudaEventRecord(kernel_stop);
    cudaEventSynchronize(kernel_stop);
    cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop);

    // D2H
    cudaEventRecord(d2h_start);
    cudaMemcpy(h_output, d_output, N * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaEventRecord(d2h_stop);
    cudaEventSynchronize(d2h_stop);
    cudaEventElapsedTime(&d2h_time, d2h_start, d2h_stop);

    // Free
    cudaEventRecord(free_start);
    cudaFree(d_output);
    cudaEventRecord(free_stop);
    cudaEventSynchronize(free_stop);
    cudaEventElapsedTime(&free_time, free_start, free_stop);

    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    cudaEventElapsedTime(&total_time, total_start, total_stop);

    // Output results
    printf("N = %d | GPU Kernel = %.4f ms | GPU Total = %.4f ms | CPU = %.4f ms\n",
           N, kernel_time, total_time, cpu_time);

    fprintf(fp, "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
            N, malloc_time, h2d_time, kernel_time, d2h_time, free_time, total_time, cpu_time);

    fclose(fp);
    free(cpu_output);
    free(h_output);

    cudaEventDestroy(malloc_start);
    cudaEventDestroy(malloc_stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    cudaEventDestroy(d2h_start);
    cudaEventDestroy(d2h_stop);
    cudaEventDestroy(free_start);
    cudaEventDestroy(free_stop);
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);

    return 0;
}
