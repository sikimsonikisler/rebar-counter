"""Make python/ and tests/ importable, and hand every test a fresh config."""

import copy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from count import load_config  # noqa: E402

_CONFIG = load_config(REPO_ROOT / "config.yaml")


@pytest.fixture
def cfg():
    """A deep copy, so a test that tweaks a threshold cannot leak into the next one."""
    return copy.deepcopy(_CONFIG)
