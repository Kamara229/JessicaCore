"""Observe frozen hidden states and compare teacher PPL proxy and EOS (not quality)."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F


@contextmanager
def capture_layers(model):
    states, hooks = {}, []
    def catch(name):
        def handler(_module, _inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(value, torch.Tensor) or value.ndim != 3:
                raise ValueError("unexpected decoder activation")
            states[name] = value.detach()[0, -1, :].float().clone()
        return handler
    try:
        hooks.append(model.model.embed_tokens.register_forward_hook(catch("embeddings")))
        for idx, block in enumerate(model.model.layers):
            hooks.append(block.register_forward_hook(catch(f"layer_{idx}")))
        yield states
    finally:
        for hook in hooks:
            hook.remove()


@torch.inference_mode()
def capture(model, token_ids):
    with capture_layers(model) as states:
        out = model(input_ids=token_ids, use_cache=False).logits[0, -1].float().clone()
    if len(states) != 31 or any(not v.isfinite().all() for v in states.values()):
        raise ValueError("unavailable or nonfinite layer activations")
    if not out.isfinite().all():
        raise ValueError("nonfinite final logits")
    return states, out


def logit_margin(logits, eos_id):
    clone = logits.clone()
    clone[eos_id] = -torch.inf
    return float(logits[eos_id] - clone.max())


@torch.inference_mode()
def compare(reference, student, teacher_logits, student_logits, eos):
    keys = ["embeddings"] + [f"layer_{i}" for i in range(30)]
    if set(reference) != set(keys) or set(student) != set(keys):
        raise ValueError("missing layer in one trajectory")
    rows = []
    for key in keys:
        teacher = reference[key].flatten().to(torch.float64)
        actual = student[key].flatten().to(torch.float64)
        if teacher.shape != actual.shape or teacher.numel() < 16:
            raise ValueError("student-teacher hidden geometry mismatch")
        rel = torch.linalg.vector_norm(teacher-actual) / torch.linalg.vector_norm(teacher).clamp_min(1e-9)
        cosine = F.cosine_similarity(teacher[None, :],actual[None, :])
        rows.append({"layer":key, "relative_L2":float(rel), "cosine":float(cosine)})
    teacher_logits=teacher_logits.to(torch.float64)
    student_logits=student_logits.to(torch.float64)
    p=F.softmax(teacher_logits,dim=-1)
    kl=float((p*(F.log_softmax(teacher_logits,dim=-1)-
                 F.log_softmax(student_logits,dim=-1))).sum())
    if kl < -1e-6: raise ValueError("negative KL")
    return {
        "layer_trajectory":rows, "teacher_to_student_KL":max(0.0,kl),
        "argmax_equal":int(teacher_logits.argmax())==int(student_logits.argmax()),
        "teacher_first_EOS":int(teacher_logits.argmax())==eos,
        "student_first_EOS":int(student_logits.argmax())==eos,
        "teacher_EOS_margin":logit_margin(teacher_logits,eos),
        "student_EOS_margin":logit_margin(student_logits,eos),
        "max_increase_layer":max(range(30),key=lambda i:
            rows[i+1]["relative_L2"]-rows[i]["relative_L2"]),
    }


def aggregate(arms):
    expected = ("CB59","v119_all","v119_no_0to4","v119_green","v119_no_0to4_green")
    if tuple(arms)!=expected: raise ValueError("incorrect arm ordering")
    answer={}
    for key,records in arms.items():
        if len(records)!=16: raise ValueError("unpaired trajectory count")
        answer[key]={
            "teacher_to_student_KL_mean":sum(r["teacher_to_student_KL"] for r in records)/16,
            "teacher_argmax_agreements":sum(r["argmax_equal"] for r in records),
            "first_EOS":sum(r["student_first_EOS"] for r in records),
            "first_EOS_logit_margin_mean":sum(r["student_EOS_margin"] for r in records)/16,
            "mean_relative_L2_by_layer":[{
                "layer":records[0]["layer_trajectory"][i]["layer"],
                "relative_L2":sum(x["layer_trajectory"][i]["relative_L2"] for x in records)/16}
                for i in range(31)],
        }
    return {"arms":answer,
            "diagnostic_only":"No semantic proof; physical archive unchanged; 16 pre-existing paired questions."}
