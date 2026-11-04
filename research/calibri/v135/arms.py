"""Isolated diagnostic replacement, never a deployable 0.7-bpp architecture."""
from contextlib import contextmanager
from v132.protocol import band_of,verify_roles

ORACLE_ARMS = (
    "CB59", "CB59_FP32_input", "CB59_FP32_head",
    "CB59_FP32_input_and_head", "v119_neutral",
    "v119_FP32_input", "v119_no_0to4", "v119_no_0to4_FP32_input"
)
BASE_ARMS=ORACLE_ARMS[:4]
PAID_ARMS=ORACLE_ARMS[4:]


@contextmanager
def oracle_modules(student, teacher, *, input=False, head=False):
    """Temporarily borrow REAL FP32 trained matrices for *diagnostic* probes.

    This reintroduces ~100MB-scale dense matrices at runtime, intentionally.
    Its quality/speed must NEVER be advertised as meeting 0.7bpp.
    """
    old_input=student.model.embed_tokens
    old_head=student.lm_head
    try:
        if input:
            student.model.embed_tokens=teacher.model.embed_tokens
        if head:
            student.lm_head=teacher.lm_head
        yield
    finally:
        student.model.embed_tokens=old_input
        student.lm_head=old_head


def set_paid(bridge,arm):
    if arm not in PAID_ARMS:
        raise ValueError("unknown paid arm")
    verify_roles(bridge)
    bridge.control(None,"not used without Green")
    if arm.startswith("v119_no_0to4"):
        for path,block in bridge.blocks.items():
            if band_of(path)==0:
                block.gain=0.
    return all(block.gain==0 for path,block in bridge.blocks.items()
               if band_of(path)==0) if arm.startswith("v119_no_0to4") else True
