"""
Device selection shared by the inference entry points.

Inference is written for a GPU. It still runs on CPU, so rather than refusing,
resolve_device() says what it found and asks before spending the time. Batch
jobs and pipelines have no one to ask, so there it stops and names the flag that
overrides it.
"""

import sys

import torch


def resolve_device(allow_cpu=False):
    """Return the torch device to run on, confirming first if it is the CPU.

    allow_cpu skips the question, for SLURM jobs and scripted runs.
    Exits rather than proceeding unconfirmed.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    print("No CUDA device found; this will run on CPU.")

    if allow_cpu:
        return torch.device("cpu")

    if not sys.stdin.isatty():
        print("Re-run with --allow-cpu to proceed.")
        raise SystemExit(1)

    try:
        answer = input("Continue on CPU? [y/N]: ").strip().lower()
    except EOFError:
        answer = ""

    if answer not in ("y", "yes"):
        raise SystemExit(1)

    return torch.device("cpu")
