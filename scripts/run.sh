#!/usr/bin/env bash
# Run the full sweep (resume-safe), then pack the results tarball.
# Args pass through to run_sweep.py (e.g. --max-workers 12).
set -uo pipefail
cd "$(dirname "$0")/.."

.venv/bin/python scripts/run_sweep.py "$@"
rc=$?

# Pack whatever completed, even if some runs failed.
scripts/pack_results.sh
exit "$rc"
