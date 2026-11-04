"""Reusable bridge facade: no physical file, learned weight or tensor is altered.

This ablates only the *runtime gain* of the first 5 layer corrections,
exactly the fixed band that reversed 12/16 first-token EOS in v132.
"""
from v132.protocol import band_of, verify_roles

ARMS = (
    "FP32", "CB59", "v119_neutral", "v119_green_current",
    "v119_drop_band0", "v119_drop_band0_green_current",
    "v119_drop_band0_green_swapped",
)


def set_arm(bridge, controller, question, opposite_question, arm):
    if arm not in ARMS[2:]:
        raise ValueError("arm does not use a bridge")
    policy = controller if "green_" in arm else None
    control = opposite_question if arm.endswith("_swapped") else question
    bridge.control(policy, control)
    verify_roles(bridge)
    if "drop_band0" in arm:
        for role, block in bridge.blocks.items():
            if band_of(role) == 0:
                block.gain = 0.0
    return {role:float(block.gain) for role, block in bridge.blocks.items()}


def assert_band0_gains(gains, arm):
    if set(gains) == set() or len(gains) != 60:
        raise ValueError("incomplete bridge")
    if "drop_band0" in arm:
        for role, gain in gains.items():
            if band_of(role) == 0 and gain != 0.0:
                raise ValueError("leaking band0 residual")
            if band_of(role) != 0 and not 0 < gain < 2:
                raise ValueError("unexpected out-of-band residual")
