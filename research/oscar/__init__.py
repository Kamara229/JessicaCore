"""Oscar standalone research FP32 -> CALIBRI compressor (no NRA)."""
from .encoder import OscarResult, encode_fp32, to_cb61
__all__ = ["OscarResult","encode_fp32","to_cb61"]
