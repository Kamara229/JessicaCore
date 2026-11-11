"""S139 actual Apple MPS post-Adam step smoke, not a model quality test."""
import json
import torch
from calibri2.s139.optimizer_aware_protection import correct_adam_stage2_displacement


def main():
    assert torch.backends.mps.is_available(), "S139 requires actual Apple MPS"
    device=torch.device("mps")
    p=torch.nn.Parameter(torch.zeros((12,320,24),device=device))
    opt=torch.optim.Adam([p],lr=0.015)
    old_grad=torch.zeros_like(p)
    old_grad[:,:,0]=1.
    attempted=torch.zeros_like(p)
    attempted[:,:,0]=-1.
    attempted[:,:,1]=1.
    p.grad=attempted
    before=p.detach().clone()
    opt.step()
    raw=p.detach().clone()
    assert float(torch.sum(old_grad*(raw-before)))>0
    safe,audit=correct_adam_stage2_displacement(
        before,raw,[old_grad],passes=32
    )
    torch.mps.synchronize()
    assert safe.device.type=="mps"
    assert float(torch.sum(old_grad*(safe-before)))<=1e-5
    assert float(torch.linalg.vector_norm(safe-before))>0
    assert audit["post_adam_projection_required"]
    assert not audit["post_adam_zero_fallback"]
    print(json.dumps({
        "event":"S139_ACTUAL_APPLE_MPS_POST_ADAM_FIRST_ORDER_SMOKE_SUCCESS",
        "projection_required":audit["post_adam_projection_required"],
        "remaining_protected_cosine":audit["post_adam_min_cosine_after"],
        "safety_method":"first_order_only",
        "physical_456_plus128_J7_tested":False,
        "model_accepted":False,
        "no_heldout_QA_no_LIVE":True,
    }),flush=True)


if __name__=="__main__":
    main()
