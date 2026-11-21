"""S87 isolated real-MPS regression: rejected Adam STEP must be atomic."""
import copy
import torch
import calibri2.s87.internal_green_tunnel as joint

assert torch.backends.mps.is_available(), "Real Apple MPS required"
device = "mps"
core = torch.nn.Parameter(torch.tensor([0.25, -0.5], device=device, dtype=torch.float16))
tunnel = joint.IntegratedGreenTunnel().to(device)
params = [core, *tunnel.parameters()]
optimizer = torch.optim.Adam(params, lr=0.0005)

class PaidSource:
    def __init__(self, centroids):
        self.centroids = centroids

sources = {}
selected = {"paid": PaidSource(core)}

# Isolate transactional behavior without loading model checkpoints or sealed QA.
original_loss = joint.teacher_forced_train_loss
original_clear = joint.clear
def synthetic_loss(*_args, **_kwargs):
    return core.float().sum() + sum(p.float().sum() for p in tunnel.parameters())
joint.teacher_forced_train_loss = synthetic_loss
joint.clear = lambda _sources: None

def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype
        assert a.device == b.device and torch.equal(a, b), "Tensor rollback mismatch"
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    else:
        assert a == b

try:
    # Populate Adam exp_avg/exp_avg_sq/step before the deliberately rejected step.
    joint.joint_train_step(None, sources, tunnel, selected, None,
                           [("TRAIN-only synthetic", "synthetic")], optimizer,
                           post_step_guard=lambda: None)
    before_params = [p.detach().clone() for p in params]
    before_adam = copy.deepcopy(optimizer.state_dict())
    calls = [0]
    def reject():
        calls[0] += 1
        raise RuntimeError("INTENTIONAL_POST_STEP_REJECTION")
    try:
        joint.joint_train_step(None, sources, tunnel, selected, None,
                               [("TRAIN-only synthetic", "synthetic")], optimizer,
                               post_step_guard=reject,
                               require_deep_tunnel_gradients=True)
    except RuntimeError as exc:
        assert str(exc) == "INTENTIONAL_POST_STEP_REJECTION"
    else:
        raise AssertionError("Rejected optimizer step was accepted")
    assert calls == [1]
    for before, after in zip(before_params, params):
        assert_equal(before, after.detach())
    assert_equal(before_adam, optimizer.state_dict())
    print("S87_REAL_MPS_ADAM_FORCED_ROLLBACK_EXACT_PASS", flush=True)
finally:
    joint.teacher_forced_train_loss = original_loss
    joint.clear = original_clear
