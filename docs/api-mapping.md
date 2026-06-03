# CUDA to HIP API Mapping Reference

## Memory Management

| CUDA | HIP |
|:---|:---|
| cudaMalloc | hipMalloc |
| cudaFree | hipFree |
| cudaMemcpy | hipMemcpy |
| cudaMemcpyAsync | hipMemcpyAsync |
| cudaMemset | hipMemset |
| cudaMemsetAsync | hipMemsetAsync |

## Device Management

| CUDA | HIP |
|:---|:---|
| cudaGetDevice | hipGetDevice |
| cudaSetDevice | hipSetDevice |
| cudaGetDeviceCount | hipGetDeviceCount |
| cudaGetDeviceProperties | hipGetDeviceProperties |

## Error Handling

| CUDA | HIP |
|:---|:---|
| cudaGetLastError | hipGetLastError |
| cudaGetErrorString | hipGetErrorString |
| cudaSuccess | hipSuccess |

## Synchronization

| CUDA | HIP |
|:---|:---|
| cudaDeviceSynchronize | hipDeviceSynchronize |
| cudaStreamSynchronize | hipStreamSynchronize |
