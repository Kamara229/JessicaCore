"""Public facade of operator rewrite experiment.

The domain-specific parts are composed as:
v84 facade -> pipeline -> archive / fit / sources / valid.
"""
from .pipeline import run
__all__ = ["run"]
