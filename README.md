# cuda2hip-porting-kit

A practical CUDA to HIP porting toolkit with small, focused examples.

## Overview

This repository documents a straightforward approach to porting CUDA workloads to HIP.

It includes:
- a CUDA baseline example
- an equivalent HIP version
- a minimal Makefile
- a small porting helper script
- notes on common migration steps

## Key features

- explicit CUDA and HIP side-by-side examples
- small build workflow using `nvcc` and `hipcc`
- lightweight automation for basic code migration
- technical notes useful for GPU workload porting

## Repository layout

```
cuda2hip-porting-kit/
├── README.md
├── LICENSE
├── .gitignore
├── Makefile
├── docs/
│   ├── porting-notes.md
│   └── design.md
├── examples/
│   ├── saxpy.cu
│   └── saxpy.cpp
└── scripts/
    └── port.py
```

## Quick start

### Build CUDA example

```bash
make cuda
```

### Build HIP example

```bash
make hip
```

### Generate HIP source from CUDA source

```bash
python scripts/port.py examples/saxpy.cu examples/saxpy.cpp
```

## Why this project exists

This repository is intended as a technical notebook for migrating small CUDA workloads toward HIP.

## License

MIT
