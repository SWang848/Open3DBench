from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PLACE3D_ROOT = REPO_ROOT / "Place-3D"
PLACE3D_INSTALL_ROOT = PLACE3D_ROOT / "install"
DREAMPLACE_ROOT = PLACE3D_INSTALL_ROOT / "dreamplace"

for path in (REPO_ROOT, PLACE3D_INSTALL_ROOT, DREAMPLACE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import Params
import PlaceDB

__all__ = ["REPO_ROOT", "PLACE3D_ROOT", "Params", "PlaceDB"]