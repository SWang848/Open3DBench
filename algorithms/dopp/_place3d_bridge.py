from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PLACE3D_ROOT = REPO_ROOT / "Place-3D"

if str(PLACE3D_ROOT) not in sys.path:
    sys.path.insert(0, str(PLACE3D_ROOT))

from dreamplace import Params, PlaceDB

__all__ = ["REPO_ROOT", "PLACE3D_ROOT", "Params", "PlaceDB"]
