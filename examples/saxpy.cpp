#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *h_x = (float*)malloc(bytes);
    float *h_y = (float*)malloc(bytes);

    for (int i = 0; i < N; ++i) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }

    float *d_x, *d_y;
    hipMalloc(&d_x, bytes);
    hipMalloc(&d_y, bytes);
    hipMemcpy(d_x, h_x, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_y, h_y, bytes, hipMemcpyHostToDevice);

    int block = 256;
    int grid = (N + block - 1) / block;
    hipLaunchKernelGGL(saxpy, dim3(grid), dim3(block), 0, 0, N, 2.0f, d_x, d_y);
    hipDeviceSynchronize();

    hipMemcpy(h_y, d_y, bytes, hipMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (h_y[i] != 4.0f) {
            ok = false;
            break;
        }
    }

    printf("[saxpy] result=%s\n", ok ? "PASS" : "FAIL");

    hipFree(d_x);
    hipFree(d_y);
    free(h_x);
    free(h_y);
    return ok ? 0 : 1;
}
