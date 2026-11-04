"""Nonnegotiable anchors from the real published Calibri v59 benchmark."""
import math
# Pinned, independently audited by v109/v110/v111 (no external imports).
ORIGINAL_CB59_SHA="50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c"
SOURCE_BYTES=11_229_303

TOTAL_UNIQUE=134_515_008
MAX_BPP=.700
MAX_SOURCE_BYTES=math.floor(MAX_BPP*TOTAL_UNIQUE/8)
RANK=2
LAYER_COUNT=30
HIDDEN=576
INTERMEDIATE=1536
HEADER=b"CLB117\x00\x00"
assert SOURCE_BYTES==11_229_303
assert MAX_SOURCE_BYTES==11_770_063
assert MAX_SOURCE_BYTES-SOURCE_BYTES==540_760

def roles():
    """Two complementary block-local roles; every index is explicitly paid."""
    result=[]
    for layer in range(LAYER_COUNT):
        result.append((f"model.layers.{layer}.self_attn.o_proj",HIDDEN,HIDDEN))
        result.append((f"model.layers.{layer}.mlp.down_proj",HIDDEN,INTERMEDIATE))
    return tuple(result)

def declared_factor_bytes():
    return sum((rows*RANK+RANK*cols)*2 for _,rows,cols in roles())

assert declared_factor_bytes()==391_680
assert declared_factor_bytes()<MAX_SOURCE_BYTES-SOURCE_BYTES
