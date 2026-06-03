# Common Porting Patterns

## Pattern 1: Simple kernel launch

CUDA: `kernel<<<N/256, 256>>>(args);`
HIP: `hipLaunchKernelGGL(kernel, dim3(N/256), dim3(256), 0, 0, args);`

## Pattern 2: Memory copy

Both use same enum values for direction.

## Pattern 3: Device query

`cudaGetDeviceProperties` -> `hipGetDeviceProperties`

## Pattern 4: Event timing

`cudaEventCreate` -> `hipEventCreate`
`cudaEventRecord` -> `hipEventRecord`
`cudaEventSynchronize` -> `hipEventSynchronize`
