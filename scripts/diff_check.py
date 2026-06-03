#!/usr/bin/env python3
"""Compare CUDA and HIP program outputs for validation."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def run_binary(path: str, timeout: int = 10) -> tuple[int, str]:
    try:
        r = subprocess.run([path], capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip()
    except Exception as e:
        return -1, str(e)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diff CUDA vs HIP output")
    parser.add_argument("cuda_bin", help="CUDA binary path")
    parser.add_argument("hip_bin", help="HIP binary path")
    args = parser.parse_args()

    rc1, out1 = run_binary(args.cuda_bin)
    rc2, out2 = run_binary(args.hip_bin)

    console.print(f"[bold]CUDA ({args.cuda_bin}):[/bold] rc={rc1}")
    console.print(out1)
    console.print(f"\n[bold]HIP ({args.hip_bin}):[/bold] rc={rc2}")
    console.print(out2)

    if rc1 == rc2 and out1 == out2:
        console.print("\n[green]MATCH -- outputs are identical[/green]")
    else:
        console.print("\n[red]MISMATCH -- outputs differ[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
