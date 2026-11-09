"""Self-contained CPU numerical sanity for S92 gradient safety; no model weights."""
import torch
from calibri2.s92.gradient_safety import project_primary_gradient


def test_halfspace_intersection():
    candidate = [torch.tensor([2.0, -1.0])]
    protection = [[torch.tensor([1.0, 0.0])], [torch.tensor([0.0, 1.0])]]
    out, diag = project_primary_gradient(candidate, protection)
    assert diag["projection_feasible"]
    assert torch.allclose(out[0], torch.tensor([2.0, 0.0]), atol=1e-5)
    for p in protection:
        assert torch.dot(out[0], p[0]) >= -1e-5


def test_protected_opposite_to_improvement():
    out, diag = project_primary_gradient(
        [torch.tensor([-1.0, 0.0])],
        [[torch.tensor([1.0, 0.0])]],
    )
    assert diag["projection_feasible"]
    assert torch.allclose(out[0], torch.zeros(2), atol=1e-5)
    assert not diag["can_apply_gradient"]


def test_invalid_shapes_and_nonfinite():
    try:
        project_primary_gradient([torch.ones(2)], [[torch.ones(3)]])
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched shape accepted")
    try:
        project_primary_gradient([torch.tensor([float("nan")])],
                                 [[torch.ones(1)]])
    except ValueError:
        pass
    else:
        raise AssertionError("Nonfinite gradient accepted")


if __name__ == "__main__":
    test_halfspace_intersection()
    test_protected_opposite_to_improvement()
    test_invalid_shapes_and_nonfinite()
    print("S92_GRADIENT_HALFSPACE_CPU_SANITY_PASS")
