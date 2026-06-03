# CUDA to HIP Porting Guide

## Overview

This guide covers the most common patterns when porting CUDA code to HIP.

## Step 1: Replace headers

```cpp
// CUDA
#include <cuda_runtime.h>

// HIP
#include <hip/hip_runtime.h>
```

## Step 2: Replace memory management

All `cudaMalloc`, `cudaFree`, `cudaMemcpy` map 1:1 to `hipMalloc`, `hipFree`, `hipMemcpy`.

## Step 3: Replace kernel launches

```cpp
// CUDA
myKernel<<<grid, block>>>(args);

// HIP
hipLaunchKernelGGL(myKernel, dim3(grid), dim3(block), 0, 0, args);
```

## Step 4: Replace synchronization

`cudaDeviceSynchronize` -> `hipDeviceSynchronize`

## Step 5: Replace error handling

`cudaGetLastError` -> `hipGetLastError`
`cudaGetErrorString` -> `hipGetErrorString`

## Common pitfalls

- HIP requires explicit `dim3()` wrapping for grid/block
- Some CUDA intrinsics have different HIP equivalents
- Texture memory API differs significantly
- `__shfl_*` intrinsics may need adjustment
