"""Mid-level STUDY facade: pinned TRAIN inputs, descriptive aggregation."""
from .sources import TRAIN_DOMAINS, WINDOWS_PER_DOMAIN, original_packed_source, train_windows
from .scoring import summarize

__all__ = ["TRAIN_DOMAINS", "WINDOWS_PER_DOMAIN",
           "original_packed_source", "train_windows", "summarize"]
