"""python scripts/sweeps <list|status|show|run> ... (see engine.py)"""
import sys
from pathlib import Path

if not __package__:
    # Run as `python scripts/sweeps`: import the package from scripts/, which
    # also makes load_results importable, as in the notebooks.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from sweeps.engine import main
else:
    from .engine import main

sys.exit(main())
