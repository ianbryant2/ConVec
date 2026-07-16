#!/usr/bin/env bash
# Pack the per-run sacred json records (not cout.txt or sweep logs) into
# sweep_results_<timestamp>.tar.gz in the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$PWD/sweep_results_${STAMP}.tar.gz"

cd epymarl/results
find sacred -type f \( -name config.json -o -name run.json -o -name metrics.json \) \
    | tar -czf "$OUT" -T -
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
