"""Physical budget for 5-bit base CB59 + one tied-embedding residual bit."""
import math
from v117.protocol import MAX_SOURCE_BYTES,TOTAL_UNIQUE,ORIGINAL_CB59_SHA
from v137.protocol import FIT_GROUPS,VALID_GROUPS,select,groups

TARGET_ROWS=49_152
TARGET_COLUMNS=576
VECTOR=8
K=32
RESIDUAL_CODEBOOK_BYTES=K*2*VECTOR*2 # two FP16 8-vectors per original K32
RESIDUAL_BITS=(TARGET_ROWS*TARGET_COLUMNS)//VECTOR
RESIDUAL_BITPLANE_BYTES=(RESIDUAL_BITS+7)//8
# Reserve 128 bytes for a versioned format with CRC32, SHA256 and checksums.
METADATA_AND_DIGEST_ALLOWANCE=128
assert RESIDUAL_BITS==3_538_944
assert RESIDUAL_BITPLANE_BYTES==442_368
assert RESIDUAL_CODEBOOK_BYTES==1024


def budget(base_bytes):
    physical=base_bytes+RESIDUAL_BITPLANE_BYTES+RESIDUAL_CODEBOOK_BYTES+METADATA_AND_DIGEST_ALLOWANCE
    if physical>MAX_SOURCE_BYTES:
        raise ValueError("cannot add a tied embedding refinement bit at <=0.7 bpp")
    return {
        "original_complete_CB59_bytes":base_bytes,
        "additional_onebit_plane_bytes":RESIDUAL_BITPLANE_BYTES,
        "additional_K32x2xV8_fp16_codebook_bytes":RESIDUAL_CODEBOOK_BYTES,
        "reserved_headers_footer_bytes":METADATA_AND_DIGEST_ALLOWANCE,
        "total_conservative_bytes":physical,
        "all_in_bpp_conservative":8*physical/TOTAL_UNIQUE,
        "remaining_headroom_bytes":MAX_SOURCE_BYTES-physical,
        "does_not_include_v119_paid_body_or_head_adapters":True,
    }
