#!/usr/bin/env python3
"""Merge sweep_results_*.tar.gz tarballs into epymarl/results/sacred.

Incoming runs are renumbered so they never clobber local ones; already-merged
runs are skipped, so rerunning is harmless.
"""

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

DEFAULT_DEST = Path(__file__).resolve().parent.parent / "epymarl" / "results" / "sacred"


def run_signature(run_dir):
    """Identity of a run for duplicate detection, or None if unreadable."""
    try:
        run = json.loads((run_dir / "run.json").read_text())
        config = json.loads((run_dir / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return (run.get("start_time"), config.get("name"), config.get("seed"))


def merge(src_sacred, dest_sacred):
    copied = skipped = 0
    for run_dir in sorted(src_sacred.glob("*/*/*")):
        if not run_dir.name.isdigit():
            continue
        dest_parent = dest_sacred / run_dir.parent.relative_to(src_sacred)
        dest_parent.mkdir(parents=True, exist_ok=True)
        existing = [d for d in dest_parent.iterdir() if d.name.isdigit()]
        sig = run_signature(run_dir)
        if sig is not None and sig in {run_signature(d) for d in existing}:
            skipped += 1
            continue
        dest = dest_parent / str(max((int(d.name) for d in existing), default=0) + 1)
        shutil.copytree(run_dir, dest)
        copied += 1
    return copied, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tarballs", nargs="+", type=Path,
                        help="sweep_results_*.tar.gz files copied back from the server")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help=f"sacred results dir to merge into (default: {DEFAULT_DEST})")
    args = parser.parse_args()

    total_copied = total_skipped = 0
    for tarball in args.tarballs:
        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(tarball) as tar:
                tar.extractall(tmp, filter="data")
            copied, skipped = merge(Path(tmp) / "sacred", args.dest)
        print(f"{tarball.name}: merged {copied} runs, skipped {skipped} already present")
        total_copied += copied
        total_skipped += skipped
    print(f"done: {total_copied} runs added to {args.dest}")


if __name__ == "__main__":
    main()
