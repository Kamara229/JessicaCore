"""Public CALIBRI v83 facade (nested study/measure facades are private).

from v83 import run   # callable entrypoint; no model is loaded on import.
"""
from .pipeline import run

__all__ = ["run"]
