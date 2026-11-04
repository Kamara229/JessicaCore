"""Per-example GOLD-ORACLE full-answer reachability through packed native C.

We explicitly optimize every correct ANSWER token plus EOS, rather than
one leading-space token. This is NOT a deployable controller; the original
real autonomous answer may remain wrong despite oracle training success.
"""
import time
import torch
from torch.nn import functional as F
from .protocol import SEED,STEPS,LEARNING_RATE
from .actuation import Actuator
from .targets import score,answer,teacher_forced_logits

def optimize(model,tok,bridge,source,prompt,ref,mode):
    torch.manual_seed(SEED)
    n=6 if mode=="shared_six_oracle" else 60
    parameters=torch.nn.Parameter(torch.zeros(n,dtype=torch.float32))
    opt=torch.optim.Adam([parameters],lr=LEARNING_RATE)
    actuator=Actuator(bridge)
    trace=[]
    start=time.perf_counter()
    for step in range(STEPS):
        gains=actuator.oracle(parameters,mode)
        logits,truth=teacher_forced_logits(model,source,ref)
        # The loss now includes ALL factual content tokens and the end marker.
        gold_loss=F.cross_entropy(logits,truth)
        total=gold_loss+.02*(gains-1).square().mean()
        opt.zero_grad(set_to_none=True)
        total.backward()
        if parameters.grad is None or not torch.isfinite(parameters.grad).all():
            raise RuntimeError("no valid native packed backward through full-answer loss")
        grad=float(parameters.grad.norm().detach())
        if grad==0:
            raise RuntimeError("no causal sensitivity of paid native residual to true answer")
        torch.nn.utils.clip_grad_norm_([parameters],3.)
        opt.step()
        if step in (0,STEPS//2,STEPS-1):
            trace.append({"step":step,
                "ORACLE_full_answer_CONTENT_and_EOS_teacher_forced_loss":float(gold_loss.detach()),
                "gradient_norm":grad,
                "gain_min":float(gains.min().detach()),
                "gain_max":float(gains.max().detach())})
    fixed=parameters.detach().clone()
    with torch.no_grad():
        gains=actuator.oracle(fixed,mode)
        first_lexical=score(model,source,ref)
        full=answer(model,tok,prompt)
    return {"parameters_after_ORACLE_FIT":fixed.tolist(),
            "oracle_train_seconds_UNUSABLE_as_regular_inference":time.perf_counter()-start,
            "gold_used_for_EVERY_answer_token_AND_EOS":True,
            "oracle_is_not_real_autonomous_green":True,
            "real_native_C_forward_backward":True,
            "native_operator_count":len(gains),
            "num_action_variables":n,
            "gain_min":float(gains.min()),
            "gain_max":float(gains.max()),
            "first_lexical_content_diagnostic":first_lexical,
            "real_greedy_output_after_oracle":full,
            "teacher_forced_TRACE_not_quality":trace}
