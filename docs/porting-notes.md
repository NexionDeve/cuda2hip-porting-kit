# Porting Notes

## Common CUDA to HIP changes

- replace `__global__` usage with HIP launch patterns where needed
- replace CUDA runtime headers with HIP headers
- update memory management calls
- update device query and error handling
- test with both CPU and GPU runs where possible

## Workflow

1. start from a small CUDA example
2. create an equivalent HIP version
3. compare behavior and runtime
4. document differences and pitfalls
