# cuda2hip-porting-kit

A small collection of CUDA to HIP porting examples with a repeatable workflow.

## Overview

This repository documents a practical CUDA to HIP porting approach.

It includes:
- a CUDA baseline example
- an equivalent HIP version
- a minimal Makefile
- a small porting helper script
- notes on common migration points

## Goals

- make porting steps explicit
- keep examples small and readable
- support local builds with CUDA and HIP toolchains
- create a foundation for future GPU porting experiments

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

## Notes

This repo is intended as a technical notebook for porting CUDA workloads to HIP.
