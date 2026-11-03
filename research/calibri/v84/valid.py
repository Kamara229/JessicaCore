"""Independent CALIBRI v84 TRAIN-VALID causal operator tests.

Full-vocabulary teacher KL/NLL here is on *TRAIN split validation slices*,
not any heldout official TEST. Uses the actual native C packed inference.
"""
from __future__ import annotations
from types import SimpleNamespace
import torch
from direct_codebook_v59 import CB59
from direct_runtime_v59 import PackedLinear
from .archive import rewrite_codebooks


@torch.inference_mode()
def score_native_runtime(packed, samples, original: bytes, changes: dict) -> dict:
    """Physical source patch -> integrity parse -> native runtime -> VALID KL."""
    source = rewrite_codebooks(original, changes) if changes else original
    decoded = CB59(source)
    old_modules = {}
    try:
        for path in changes:
            m = packed.get_submodule(path)
            if not isinstance(m, PackedLinear):
                raise AssertionError("source-trained native operator no longer packed")
            old_modules[path] = m.handle
            m.handle = decoded.handles[path + ".weight"]
        domains = {}
        for domain in ("wiki_valid", "news_valid"):
            kl, nll, teacher_nll = [], [], []
            for item in samples[domain]["windows"]:
                ids = item["tokens"]
                student = packed(input_ids=ids[None],use_cache=False).logits[0, :-1]
                teacher = item["teacher_logits"][:-1]
                if not (torch.isfinite(student).all() and torch.isfinite(teacher).all()):
                    raise AssertionError("nonfinite genuine native packed logits")
                # torch float64 to protect low student softmax probabilities
                a = teacher.to(torch.float64).log_softmax(-1)
                b = student.to(torch.float64).log_softmax(-1)
                p = a.exp()
                kl.append(float((p * (a - b)).sum(dim=-1).mean()))
                labels = ids[1:].long()
                nll.append(float(-b.gather(1,labels[:,None]).mean()))
                teacher_nll.append(float(-a.gather(1,labels[:,None]).mean()))
            domains[domain] = {
                "real_original_teacher_student_full_vocab_KL": sum(kl)/len(kl),
                "real_original_next_token_NLL": sum(nll)/len(nll),
                "real_original_teacher_next_token_NLL": sum(teacher_nll)/len(teacher_nll),
                "windows": len(kl),
                "positions": len(kl)*(ids.numel()-1),
            }
        return {
            "source_sha_verified_at_loading": decoded.hash(),
            "native_base_operators_are_packed": True,
            "per_train_valid_corpus": domains,
            "mean_KL": sum(x["real_original_teacher_student_full_vocab_KL"]
                           for x in domains.values())/len(domains),
        }
    finally:
        for path, h in old_modules.items():
            packed.get_submodule(path).handle = h


def choose_guarded_variant(baseline: dict, variants: dict) -> tuple[str, dict]:
    """Select on actual original TRAIN-VALID corpora; never see any TEST."""
    best_name = "baseline"
    best = baseline
    base = baseline["per_train_valid_corpus"]
    for name, result in variants.items():
        domains = result["per_train_valid_corpus"]
        # Unchanged/non-regressing across both original train-validation corpora
        safe = all(domains[d]["real_original_teacher_student_full_vocab_KL"]
                   <= base[d]["real_original_teacher_student_full_vocab_KL"] + 1e-6
                   for d in ("wiki_valid","news_valid"))
        if safe and result["mean_KL"] < best["mean_KL"] - 1e-4:
            best_name, best = name, result
    return best_name, {
        "original_baseline_mean_KL": baseline["mean_KL"],
        "selected_mean_KL": best["mean_KL"],
        "selected_variant": best_name,
        "both_VALID_domains_nonregressing": best_name != "baseline",
        "newly_rewritten_source_valid_guard_pass": best_name != "baseline",
        "not_independent_TEST_and_not_a_production_gate": True,
    }
