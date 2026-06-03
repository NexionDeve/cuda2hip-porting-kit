#!/usr/bin/env python3
import re
import sys
from pathlib import Path

CUDA_TO_HIP = [
    (r"#include\s*<cuda_runtime.h>", "#include <hip/hip_runtime.h>"),
    (r"cudaMalloc", "hipMalloc"),
    (r"cudaMemcpy\(([^,]+),\s*([^,]+),\s*([^,]+),\s*cudaMemcpyHostToDevice\)", r"hipMemcpy(\1, \2, \3, hipMemcpyHostToDevice)"),
    (r"cudaMemcpy\(([^,]+),\s*([^,]+),\s*([^,]+),\s*cudaMemcpyDeviceToHost\)", r"hipMemcpy(\1, \2, \3, hipMemcpyDeviceToHost)"),
    (r"cudaFree", "hipFree"),
    (r"cudaDeviceSynchronize", "hipDeviceSynchronize"),
    (r"__global__\s+void\s+", "__global__ void "),
]

def main():
    if len(sys.argv) != 3:
        print("Usage: port.py <input.cu> <output.cpp>")
        sys.exit(1)

    src = Path(sys.argv[1]).read_text(encoding="utf-8")
    out = src
    for pattern, repl in CUDA_TO_HIP:
        out = re.sub(pattern, repl, out)

    Path(sys.argv[2]).write_text(out, encoding="utf-8")
    print(f"[port] wrote {sys.argv[2]}")

if __name__ == "__main__":
    main()
