"""Make the ``app`` package importable when running pytest from ``backend/``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
