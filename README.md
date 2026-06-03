# cuda2hip-porting-kit

[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![HIP](https://img.shields.io/badge/HIP-ROCm%206.x-ED1C24?logo=amd&logoColor=white)](https://rocm.docs.amd.com/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Practical CUDA to HIP porting toolkit with automated migration helpers, side-by-side examples, and porting documentation.**

A structured workflow for migrating CUDA workloads to AMD's HIP framework.

---

## Architecture

```
+--------------------------------------------------------------+
|                     scripts/port.py                           |
|               (automated CUDA -> HIP migration)               |
+----------+----------+---------------+-------------------------+
|  SAXPY   | Vec Add  | Matrix Mul    | Reduction               |
|  CUDA    | CUDA     | CUDA          | CUDA                    |
|  -> HIP  | -> HIP   | -> HIP        | -> HIP                  |
+----------+----------+---------------+-------------------------+
|                    docs/                                       |
|  porting-guide.md    |  common-patterns.md                    |
|  migration checklist |  API mapping reference                 |
+--------------------------------------------------------------+
```

## Features

| Component | What it does | Details |
|:---|:---|:---|
| **Auto Port** | Automated source migration | Regex-based CUDA->HIP transforms |
| **Examples** | Side-by-side CUDA/HIP | SAXPY, vector add, matrix mul, reduction |
| **Build System** | Unified Makefile | nvcc + hipcc targets |
| **Docs** | Porting guide | Common patterns, API mapping, pitfalls |
| **Diff Analysis** | Migration validation | Compare CUDA vs HIP output |

### CUDA -> HIP Mapping

| CUDA | HIP | Notes |
|:---|:---|:---|
| `cudaMalloc` | `hipMalloc` | Direct 1:1 mapping |
| `cudaMemcpy` | `hipMemcpy` | Same enum values |
| `cudaFree` | `hipFree` | Direct 1:1 mapping |
| `<<<grid, block>>>` | `hipLaunchKernelGGL` | Different launch syntax |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` | Direct 1:1 mapping |
| `nvcc` | `hipcc` | Compiler swap |

---

## Quick Start

### Prerequisites

- CUDA Toolkit 12.x (for building CUDA examples)
- ROCm 6.x with HIP (for building HIP examples)
- Python 3.10+

### Install

```bash
git clone https://github.com/NexionDeve/cuda2hip-porting-kit.git
cd cuda2hip-porting-kit
pip install -r requirements.txt
```

### Auto-port a CUDA file

```bash
python scripts/port.py examples/saxpy.cu examples/saxpy.cpp
```

### Build examples

```bash
make cuda    # Build all CUDA examples
make hip     # Build all HIP examples
make all     # Build both
make clean   # Clean builds
```

### Run examples

```bash
./build/saxpy_cuda
./build/saxpy_hip
```

---

## Project Structure

```
cuda2hip-porting-kit/
+-- examples/
|   +-- saxpy.cu              # SAXPY CUDA implementation
|   +-- saxpy.cpp             # SAXPY HIP port
|   +-- vector_add.cu         # Vector add CUDA
|   +-- vector_add.cpp        # Vector add HIP
|   +-- matrix_mul.cu         # Matrix multiply CUDA
|   +-- matrix_mul.cpp        # Matrix multiply HIP
|   +-- reduction.cu          # Reduction CUDA
|   +-- reduction.cpp         # Reduction HIP
+-- scripts/
|   +-- port.py               # Automated porting tool
|   +-- diff_check.py         # Output comparison
+-- docs/
|   +-- porting-guide.md      # Full porting guide
|   +-- common-patterns.md    # Common migration patterns
|   +-- api-mapping.md        # CUDA->HIP API reference
+-- build/                    # Compiled binaries
+-- Makefile                  # Build system
+-- requirements.txt
+-- pyproject.toml
+-- LICENSE
```

---

## Development

```bash
pip install -e ".[dev]"
ruff check .
python scripts/port.py --dry-run examples/saxpy.cu
```

---

## Roadmap

- [ ] cuBLAS -> hipBLAS migration examples
- [ ] cuDNN -> MIOpen migration examples
- [ ] CUDA Graphs -> HIP Graphs porting
- [ ] Nsight -> rocprof profiling guide
- [ ] Automated test suite for ported code

---

## License

MIT -- see [LICENSE](LICENSE).
