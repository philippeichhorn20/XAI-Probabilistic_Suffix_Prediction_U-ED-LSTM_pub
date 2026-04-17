"""Configuration module for interpretability notebooks."""

import os
import shutil

# Ensure Homebrew binaries (e.g. graphviz `dot`) are on PATH when running
# inside an IDE that doesn't inherit the full shell environment.
if shutil.which("dot") is None:
    _brew_bin = "/opt/homebrew/bin"
    if os.path.isdir(_brew_bin):
        os.environ["PATH"] = _brew_bin + ":" + os.environ.get("PATH", "")

from .base_config import BaseConfig

__all__ = ['BaseConfig']
