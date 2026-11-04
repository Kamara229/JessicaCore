"""Ideal supervision reachability bound through genuine packed C backward.

Gold IS used to optimize EACH individual example. Therefore any gain from
these oracle controls is NOT a deployable GreenTunnel model improvement.
By comparing six-degree vs sixty-degree controls we diagnose whether
prior small action bottleneck was too restrictive.
"""
import time
import torch
from torch.nn import functional as F
from .protocol import SEED,STEPS,LEARNING_RATE
from .actuation import Actuator
from .targets import score,answer

def optimize(model,tok,bridge,source,prompt,target,other,mode):
    torch.manual_seed(SEED)
    count=6 if mode=="shared_six_oracle" else 60
    raw=torch.nn.Parameter(torch.zeros(count,dtype=torch.float32))
    optim=torch.optim.Adam([raw],lr=LEARNING_RATE)
    actuator=Actuator(bridge)
    trace=[]
    began=time.perf_counter()
    for step in range(STEPS):
        gains=actuator.oracle(raw,mode)
        logits=model(input_ids=source,use_cache=False).logits[0,-1].float()
        loss=F.cross_entropy(logits[None,:],torch.tensor([target]))
        total=loss+.02*(gains-1).square().mean()
        optim.zero_grad(set_to_none=True)
        total.backward()
        if raw.grad is None or not bool(torch.isfinite(raw.grad).all()):
            raise RuntimeError("native packed-model first-token gradient missing")
        grad_norm=float(raw.grad.norm().detach())
        if not grad_norm:
            raise RuntimeError("no causal sensitivity to gold target")
        torch.nn.utils.clip_grad_norm_([raw],3.)
        optim.step()
        if step in (0,STEPS//2,STEPS-1):
            trace.append({"step":step,
                "ORACLE_gold_first_token_fit_loss":float(loss.detach()),
                "gradient_norm":grad_norm,
                "gain_min":float(gains.min().detach()),
                "gain_max":float(gains.max().detach())})
    fixed=raw.detach().clone()
    with torch.no_grad():
        factors=actuator.oracle(fixed,mode)
        outcome=score(model,source,target,other)
        output=answer(model,tok,prompt)
    return {"parameters_after_ORACLE_FIT":fixed.tolist(),
            "train_seconds_not_deployable_inference":time.perf_counter()-began,
            "gold_used_for_example_specific_ORACLE_optimization":True,
            "real_native_gradient_path":True,
            "num_steered_native_operators":len(factors),
            "control_parameter_count":count,
            "gain_min":float(factors.min()),
            "gain_max":float(factors.max()),
            "first_token_diagnostic":outcome,
            "real_greedy_output_after_oracle":output,
            "trace":trace}
