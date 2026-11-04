"""Fixed six five-layer depth bands, not selected after seeing performance."""
from v131.protocol import examples

BANDS = tuple((start, start + 5) for start in range(0, 30, 5))
assert BANDS == ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30))


def band_of(role):
    layer = int(role.split(".")[2])
    if layer not in range(30):
        raise ValueError("bad depth")
    return layer // 5


def conditions():
    # Baseline head-only vs all paid body residuals; leave-one-out and
    # add-one-in assess necessity and sufficiency separately (not proof).
    yield "head_only", lambda b: False
    yield "complete", lambda b: True
    for i in range(len(BANDS)):
        yield f"drop_band_{i}", lambda b, i=i: b != i
        yield f"only_band_{i}", lambda b, i=i: b == i


def verify_roles(bridge):
    if len(bridge.blocks) != 60:
        raise ValueError("unexpected physical v119 paid-body role count")
    counts = [0] * len(BANDS)
    for role in bridge.blocks:
        counts[band_of(role)] += 1
    if counts != [10] * 6:
        raise ValueError("unexpected 5-layer role grouping")


def all_examples():
    rows = tuple(examples())
    if len(rows) != 16:
        raise AssertionError("unexpected fixed paired diagnostic data")
    return rows
