#include <cstdio>
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 4;
    float h_A[N] = {1.0f, 2.0f, 3.0f, 4.0f};
    // in real GPU programs, the input size N is usually not known ahead of time. so we need to use dynamic memory allocation.
    // for example, we can use cudaMalloc to allocate memory for the input arrays.
    // cudaMalloc returns a pointer to the allocated memory.
    // we can use this pointer to pass the input arrays to the kernel.
    // we can use cudaFree to free the allocated memory.
    // cudaFree returns a pointer to the freed memory.
    // we can use this pointer to pass the input arrays to the kernel.
    // we can use cudaFree to free the allocated memory.
    // cudaFree returns a pointer to the freed memory.
    // we can use this pointer to pass the input arrays to the kernel.
    // float *h_A = (float *)malloc(N * sizeof(float));
    // float *h_B = (float *)malloc(N * sizeof(float));
    // float *h_C = (float *)malloc(N * sizeof(float));
    float h_B[N] = {5.0f, 6.0f, 7.0f, 8.0f};
    float h_C[N];

    float *d_A, *d_B, *d_C;
    size_t bytes = N * sizeof(float);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    vector_add<<<1, 32>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Print result
    printf("C = [");
    for (int i = 0; i < N; i++) {
        printf("%.1f", h_C[i]);
        if (i < N - 1) printf(", ");
    }
    printf("]\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

// nvcc -g -G CUDA_0001_Vector_Addition.cu -o test
// g++ -std=c++17 xxx.cpp -g -O0 -o test
