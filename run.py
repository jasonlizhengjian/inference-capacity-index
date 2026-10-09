#!/usr/bin/env python3
"""Checkout-local entry point; frozen-data commands need only Python 3.12+."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from agent_index.cli import main

if __name__ == "__main__":
    main()
