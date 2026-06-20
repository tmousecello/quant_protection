"""Make artifacts/phase3/ importable so tests can import make_golden (the golden generator)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
