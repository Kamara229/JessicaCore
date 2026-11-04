"""Fixed no-leak probe corpus; no optimizer and no official v89 TEST."""
from v132.protocol import all_examples, band_of, verify_roles

ARMS = ("FP32", "CB59", "v119_all", "v119_no_0to4", "v119_green", "v119_no_0to4_green")
DEPTH = 30


def apply(bridge, policy, question, arm):
    if arm not in ARMS[2:]:
        raise ValueError("not a supported paid-body arm")
    verify_roles(bridge)
    bridge.control(policy if arm.endswith("green") else None, question)
    if "no_0to4" in arm:
        for name, block in bridge.blocks.items():
            if band_of(name) == 0:
                block.gain = 0.0
    if "no_0to4" in arm and any(block.gain != 0 for name, block in bridge.blocks.items()
                                if band_of(name) == 0):
        raise ValueError("band0 residual not disabled")


def corpus():
    samples = tuple(all_examples())
    if len(samples) != 16:
        raise ValueError("counterfactual corpus incomplete")
    return samples
