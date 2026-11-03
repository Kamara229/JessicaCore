"""Actual native C original-operator / conditional-operator MODEL validation.

No TEST data, no oracle dataset labels for runtime conditional routing.
"""
from __future__ import annotations
import time
from contextlib import contextmanager
import torch
from .native_ops import insert_native_adapters, restore_native_adapters


@contextmanager
def temporarily_rewrite(student,source,paths):
    previous=insert_native_adapters(student,source,tuple(paths))
    try:
        yield
    finally:
        restore_native_adapters(previous)


@torch.inference_mode()
def score(student,valid,source,paths):
    with temporarily_rewrite(student,source,paths):
        corpora={}
        for domain,part in valid.items():
            if not domain.endswith("_valid"):
                raise AssertionError("never score a TEST corpus for source selection")
            kls,nlls,times=[],[],[]
            for row in part["windows"]:
                tokens=row["tokens"]
                t0=time.perf_counter()
                y=student(input_ids=tokens[None],use_cache=False).logits[0,:-1]
                times.append(time.perf_counter()-t0)
                ref=row["teacher_logits"][:-1]
                if not torch.isfinite(y).all():
                    raise AssertionError("invalid genuine native operator inference")
                lp=ref.to(torch.float64).log_softmax(-1)
                lq=y.to(torch.float64).log_softmax(-1)
                kls.append(float((lp.exp()*(lp-lq)).sum(-1).mean()))
                nlls.append(float(-lq.gather(1,tokens[1:,None]).mean()))
            corpora[domain]={
                "real_original_teacher_student_full_vocab_KL":sum(kls)/len(kls),
                "real_next_token_NLL":sum(nlls)/len(nlls),
                "original_positions_scored":len(kls)*(len(tokens)-1),
                "CPU_full_model_seconds":sum(times)}
    return {
        "per_train_valid_corpus":corpora,
        "mean_KL":sum(p["real_original_teacher_student_full_vocab_KL"]
                      for p in corpora.values())/len(corpora),
        "adaptive_operator_paths":list(paths),
        "routes_by_runtime_activations_not_corpus_labels":True,
        "native_C_executes_both_codebook_branches":bool(paths),
        "not_fused_so_speed_claim_prohibited":True,
    }


def variant_sets(paths):
    # fixed before first v85 VALID, no adaptive TEST selection
    return [(name,(name,)) for name in paths] + [
        ("layer11_plus_layer28",(paths[0],paths[1])),
        ("layer11_plus_layer29",(paths[0],paths[2])),
        ("layer28_plus_layer29",(paths[1],paths[2])),
        ("all_three_conditional_operators",tuple(paths))]
