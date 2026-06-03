#!/usr/bin/env python3
"""Automated CUDA to HIP source migration tool."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

TRANSFORMS = [
    (r"#include\s*<cuda_runtime\.h>", "#include <hip/hip_runtime.h>"),
    (r"#include\s*<cuda_runtime\.h>", "#include <hip/hip_runtime.h>"),
    (r"cudaMalloc\b", "hipMalloc"),
    (r"cudaFree\b", "hipFree"),
    (r"cudaMemcpy\b", "hipMemcpy"),
    (r"cudaMemcpyHostToDevice", "hipMemcpyHostToDevice"),
    (r"cudaMemcpyDeviceToHost", "hipMemcpyDeviceToHost"),
    (r"cudaMemcpyDeviceToDevice", "hipMemcpyDeviceToDevice"),
    (r"cudaDeviceSynchronize", "hipDeviceSynchronize"),
    (r"cudaGetLastError", "hipGetLastError"),
    (r"cudaGetErrorString", "hipGetErrorString"),
    (r"cudaSuccess", "hipSuccess"),
]


def port_file(src: Path, dst: Path, dry_run: bool = False) -> dict:
    content = src.read_text(encoding="utf-8")
    changes = []

    for pattern, replacement in TRANSFORMS:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            changes.append((pattern, replacement))
            content = new_content

    # Convert kernel launch syntax
    launch_pattern = r"(\w+)<<<([^>]+)>>>([^;]+);"
    def convert_launch(m):
        kernel = m.group(1)
        args_block = m.group(2)
        call_args = m.group(3)
        parts = [a.strip() for a in args_block.split(",")]
        if len(parts) == 2:
            grid, block = parts
        elif len(parts) == 3:
            grid, block, _ = parts  # skip shared mem
        else:
            grid, block = parts[0], "1"
        return f"hipLaunchKernelGGL({kernel}, dim3({grid}), dim3({block}), 0, 0{call_args});"

    new_content = re.sub(launch_pattern, convert_launch, content)
    if new_content != content:
        changes.append(("<<<grid,block>>>", "hipLaunchKernelGGL"))
        content = new_content

    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content, encoding="utf-8")

    return {"src": str(src), "dst": str(dst), "changes": changes}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CUDA to HIP porting tool")
    parser.add_argument("source", help="Source CUDA file")
    parser.add_argument("output", help="Output HIP file")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    args = parser.parse_args()

    result = port_file(Path(args.source), Path(args.output), dry_run=args.dry_run)

    table = Table(title="Porting Results")
    table.add_column("Pattern")
    table.add_column("Replacement")
    for pat, repl in result["changes"]:
        table.add_row(pat, repl)

    if result["changes"]:
        console.print(table)
        console.print(f"[green]{len(result['changes'])} transforms applied[/green]")
    else:
        console.print("[yellow]No transforms needed[/yellow]")


if __name__ == "__main__":
    main()
