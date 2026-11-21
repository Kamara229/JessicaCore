"""S119q1: genuine SINGLE original 43M+GT decoder forward for on-path NRA3/NPA1.

Mechanical ABI pilot, NOT an accepted trained single-pass architecture.
Unlike S119m, NO separate full12 first-pass is used for the candidate:
layers 0..5 are captured during the SAME original causal prefix forward and
the trainable S119m core/NRA3/NPA1 child injects after original layer5.
Old S119m child expected a 12-layer tensor; six missing later slots are
explicitly replaced with COPIES of causal layer5 hidden, never fabricated
future-layer states. This is a distribution shift requiring new TRAIN-only
joint learning before any quality claim.

Six pre-registered S119n author prompts, first native token each, no GOLD.
For each prompt, separate original baseline, one-pass candidate and zero-gate
control native full 49,152-vocab forwards (3, NOT two passes per candidate).
NO optimization, original/old FP16 weights unchanged. Original full S101/J7
controller not replayed; no independent improvement, CPU or bpp gate proof.
No model deployment, main, LIVE, Render, Supabase, Jessica or APK.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import torch

from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119m_real43m_gt_interlayer_onpath_joint import (
    old_original_param_digest,
)
from research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts import (
    PROMPTS, REPORT as S119N_REPORT, load_parent, prompt_features,
    first_original_pass,
)
from research.calibri_nra3.s119p_real43m_onpath_component_ablation import (
    verify_original_research_report, temporary_gate_mask,
)
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    load_actual_original, sha,
)
from research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation import (
    strict_chat_source_ids,
)
from calibri2.s160 import real_parallel_slots_j7 as s160
from v176.operators import clear

STAGE = "CALIBRI_S119Q1_REAL43M_GT_ONEPASS_CAUSAL_EARLY_STATE_GATE0"
REPORT = "S119Q1_ONEPASS_CAUSAL_EARLY_NATIVE_FIRST_TOKEN_ONLY.json"
EARLY_LAYERS = (0, 1, 2, 3, 4, 5)
PLACEHOLDER_MAPPING = (0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5)
SEED = 119129


def build_early_causal_view(recorded: dict[int, torch.Tensor]):
    """Adapt immutable 12-slot ABI using ONLY original layers <=5 in THIS pass."""
    check(set(recorded) == set(EARLY_LAYERS),
          "exact early original layers 0..5 must exist before injection")
    states = [recorded[idx].detach() for idx in EARLY_LAYERS]
    shape = states[0].shape
    check(len(shape) == 3 and shape[0] == 1 and shape[1] >= 2
          and shape[2] == 320
          and all(s.shape == shape and bool(torch.isfinite(s).all())
                  for s in states),
          "physical causal early h320 same current prefix only")
    reused = [recorded[i].detach() for i in PLACEHOLDER_MAPPING]
    x = torch.stack(reused, dim=1)
    check(x.shape == (1,12,shape[1],320)
          and torch.equal(x[:,5],x[:,11]),
          "source-only repeated early h5, NO imagined real layers6..11")
    return x


@contextmanager
def single_pass_early_intervention(model, adapter):
    """Temporary original decoder hook; one complete forward, never replay.

    Early states and layer5 must be produced by exactly one SAME call.
    Never modify past_key_values, original decoder weights, or later hooks.
    """
    check(len(model.model.layers) == 12
          and adapter.output_gate == 0
          and not adapter.candidate_accepted
          and not adapter.activation_allowed,
          "real original12 plus unpromoted trained child only")
    original_hooks = {
        i:set(layer._forward_hooks)
        for i,layer in enumerate(model.model.layers[:6])
    }
    seen = {}
    witness = {"candidate_forward_calls": 0,
               "early_layer_calls": {},
               "adapter_delta_H320": None,
               "placeholder_mapping": list(PLACEHOLDER_MAPPING)}
    handles = []
    completed = False

    def capture(i):
        def hook(_mod, _args, output):
            check(i not in seen, "one real pass through early layer only")
            hidden = output[0] if isinstance(output,tuple) else output
            check(isinstance(hidden,torch.Tensor) and hidden.ndim==3
                  and hidden.shape[-1]==320 and bool(torch.isfinite(hidden).all()),
                  "physical original H320 early layer output")
            seen[i] = hidden.detach()
            witness["early_layer_calls"][i] = 1
            return None
        return hook

    def mid_hook(_mod, _args, output):
        check(5 not in seen and set(seen)==set(range(5)),
              "all previous early layers must execute before layer5")
        hidden = output[0] if isinstance(output,tuple) else output
        check(isinstance(hidden,torch.Tensor) and hidden.ndim==3
              and hidden.shape[0]==1 and hidden.shape[-1]==320
              and bool(torch.isfinite(hidden).all()),
              "real original layer5 returned H320")
        seen[5] = hidden.detach()
        witness["early_layer_calls"][5] = 1
        x = build_early_causal_view(seen)
        delta = adapter(x,contextual=True)
        check(delta.shape == (1,320)
              and delta.requires_grad and bool(torch.isfinite(delta).all()),
              "original native later layers see differentiable onepass child")
        witness["adapter_delta_H320"] = float(
            torch.linalg.vector_norm(delta.detach()).item())
        witness["candidate_forward_calls"] += 1
        patch = torch.cat((torch.zeros_like(hidden[:,:-1,:]),
                           delta.unsqueeze(1)),dim=1)
        injected = hidden + patch
        return (injected,*output[1:]) if isinstance(output,tuple) else injected

    try:
        for idx in range(5):
            handles.append(model.model.layers[idx].register_forward_hook(
                capture(idx)))
        handles.append(model.model.layers[5].register_forward_hook(mid_hook))
        yield witness
        completed = True
        check(witness["candidate_forward_calls"] == 1
              and set(seen) == set(EARLY_LAYERS)
              and len(witness["early_layer_calls"]) == 6,
              "single actual original decoder forward, all early causal states")
    finally:
        for handle in handles:
            handle.remove()
        check(all(set(model.model.layers[i]._forward_hooks) == original_hooks[i]
                  for i in EARLY_LAYERS),
              "temporary causal hooks MUST be removed even on failure")
        if completed:
            check(witness["candidate_forward_calls"] == 1,
                  "single-pass candidate cannot replay early decoder")


def onepass_full_native_logits(model,sources,bridge,full,expansion,adapter,
                               native_prefix,features):
    """One real frozen S142+GT forward only; adapter builds delta at layer5."""
    check(isinstance(native_prefix,list) and 2 <= len(native_prefix) < 160
          and features.shape == (512,)
          and all(isinstance(t,int) and t>=0 for t in native_prefix),
          "only current native causal prompt prefix, no future GOLD")
    ids = torch.tensor([native_prefix], dtype=torch.long)
    with torch.enable_grad(),expansion.prefill_only(model), \
         s160.prefill_only_tunnel(model,full), \
         bridge.prompt_scope("S93",features), \
         single_pass_early_intervention(model,adapter) as witness:
        clear(sources)
        out = model(input_ids=ids,past_key_values=None,
                    use_cache=True,output_hidden_states=False)
        check(out.past_key_values is not None and
              out.logits.shape == (1,len(native_prefix),49152),
              "physical original12 and full native 49152 LM head")
        logits = out.logits[:,-1,:].float()
        clear(sources)
        del out
    check(logits.requires_grad and bool(torch.isfinite(logits).all()),
          "one ORIGINAL forward must feed adapter gradient to FULL native head")
    return logits,witness


def run(s93:Path,side:Path,s142:Path,parent:Path,parent_report:Path,
        s119n_report:Path,out:Path):
    check(not out.exists() and s119n_report.name == S119N_REPORT,
          "exact original S119n six source prompts, no overwriting outputs")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    prior=json.loads(s119n_report.read_text(encoding="utf8"))
    records=verify_original_research_report(prior,s93,s142,parent)
    _,adapter,old_53751=load_parent(s93,side,s142,parent,parent_report)
    original,sources,bridge,full,expansion=load_actual_original(
        s93,side,s142)
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tokenizer.vocab_size==49152,
          "full physical original native tokenizer, no proxy classes")
    original.model.add_module("nra3_s119m_onpath",adapter)
    original.eval()
    check(all(not p.requires_grad for n,p in original.named_parameters()
              if "nra3_s119m_onpath" not in n),
          "freeze every original pretrained Core12 Green Tunnel weight")
    original_digest=old_original_param_digest(original)
    snapshot={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}
    hook_baseline={i:set(layer._forward_hooks)
                   for i,layer in enumerate(original.model.layers[:6])}
    observations=[]
    real_decoder_forwards=0
    gate_grad_norm=None

    for (group,question),past in zip(PROMPTS,records):
        prefix=strict_chat_source_ids(tokenizer,question)
        route=prompt_features(tokenizer,question)
        # REFERENCE original native CPU pass. This is NOT fed to the candidate.
        baseline,unused_reference_all12=first_original_pass(
            original,sources,bridge,full,expansion,prefix,route)
        real_decoder_forwards+=1
        check(int(baseline.argmax(-1))==
              past["original_native_greedy"]["generated_native_token_ids"][0],
              "reproduce original S119n first native top1 without labels")
        del unused_reference_all12
        # Candidate computes its own early H320 in its ONE decoder pass.
        logits,witness=onepass_full_native_logits(
            original,sources,bridge,full,expansion,adapter,prefix,route)
        real_decoder_forwards+=1
        native_first=int(logits.detach().argmax(-1).item())
        if gate_grad_norm is None:
            # Sensitivity, NOT training: score one original-native logit.
            score=logits[0,int(baseline.argmax(-1).item())]
            gradient=torch.autograd.grad(
                score,adapter.midlayer_gates,allow_unused=False)[0]
            check(gradient.shape==(3,) and
                  bool(torch.isfinite(gradient).all()) and
                  float(gradient.abs().sum().item())>1e-12,
                  "REAL original LM logit must depend on in-core neural gates")
            gate_grad_norm=float(torch.linalg.vector_norm(gradient).item())
        candidate=logits.detach().cpu().clone()
        del logits
        with temporary_gate_mask(adapter,(0,0,0)):
            zero,zero_witness=onepass_full_native_logits(
                original,sources,bridge,full,expansion,adapter,prefix,route)
            zero=zero.detach().cpu().clone()
        real_decoder_forwards+=1
        check(torch.allclose(zero,baseline,atol=2e-4,rtol=1e-5)
              and torch.equal(zero.argmax(-1),baseline.argmax(-1)),
              "ONEPASS ZERO gate must preserve physical original native logits")
        check(all(set(original.model.layers[i]._forward_hooks)==hook_baseline[i]
                  for i in EARLY_LAYERS),
              "real layers hooks were not leaked")
        old_top2=torch.topk(baseline[0],2).values
        margin=float((old_top2[0]-old_top2[1]).item())
        logit_shift=float((candidate-baseline).abs().max().item())
        observations.append({
            "group":group,
            "source_prompt_SHA256":hashlib.sha256(
                question.encode("utf8")).hexdigest(),
            "native_prefix_length":len(prefix),
            "original_native_first_id":int(baseline.argmax(-1).item()),
            "onepass_native_first_id":native_first,
            "onepass_changed_original_first_id":
                native_first != int(baseline.argmax(-1).item()),
            "original_top1_top2_margin":margin,
            "onepass_full_vocab_Linf_logit_shift":logit_shift,
            "margin_certifies_stability":margin > 2*logit_shift,
            "onepass_H320_residual_norm":witness["adapter_delta_H320"],
            "zero_gate_onepass_H320_residual_norm":
                zero_witness["adapter_delta_H320"],
            "original_native_full_vocab_ZERO_gate_parity":True,
            "onepass_actual_decoder_forward_calls":1,
            "late_layer_6_to_11_still_original_frozen":True,
            "placeholder_layers_are_copies_of_layer5_NOT_real_layers":True,
        })
    check(real_decoder_forwards==18 and len(observations)==6
          and old_original_param_digest(original)==original_digest,
          "six baseline + six ONEPASS candidate + six ONEPASS zero, frozen")
    check(all(torch.equal(snapshot[k],v.detach().cpu())
              for k,v in adapter.state_dict().items()),
          "ALL original trained S119m + NRA3/NPA1 FP16 remain exact")
    for key,v in old_53751.items():
        short=key[len("model.nra3_calibri_joint.s119m_onpath."):]
        check(torch.equal(adapter.state_dict()[short].detach().half().cpu(),v),
              "original S115 53,751 FP16 always byte-identical")
    check(not adapter.candidate_accepted and adapter.output_gate==0
          and not adapter.activation_allowed,
          "mechanical early-state prototype NEVER activated")
    result={
        "stage":STAGE,
        "source_s119n_report_SHA256":sha(s119n_report),
        "source_s119m_checkpoint_SHA256":sha(parent),
        "original_s142_43M_GT_SHA256":sha(s142),
        "original_s93_CB61_SHA256":sha(s93),
        "real_original_decoder_forward_total":real_decoder_forwards,
        "actual_single_original_forward_PER_candidate":True,
        "zero_gate_original_full_native_vocabulary_49152_parity":True,
        "original_12layer_core_GreenTunnel_and_old_53751_FP16_unmodified":True,
        "onepass_strictly_early_original_layer_indices":list(EARLY_LAYERS),
        "synthetic_later_slots_copy_from_early_layer":list(PLACEHOLDER_MAPPING),
        "not_genuine_later_original_hidden_state_representations":True,
        "old_S119m_trained_on_full12_not_early_imputed":True,
        "full_vocab_original_native_gate_sensitivity_gradient_norm":
            gate_grad_norm,
        "no_optimizer_or_training":True,
        "no_independent_quality_score_or_gold_labels":True,
        "only_six_reused_S119n_author_prompts":True,
        "no_original_S101_J7_controller_replay":True,
        "no_new_original_616_J7_EOS_cert":True,
        "no_packed_all_in_bpp_or_paired_KV_CPU_speed":True,
        "no_GOLD_S118_B_or_spent_DEV_EVAL128":True,
        "no_LIVE_Render_Supabase_Jessica_main_APK_or_deploy":True,
        "candidate_accepted":False,
        "output_gate":0,
        "research_records":observations,
        "summary":{
            "six_native_first_tokens":len(observations),
            "changed_first_ids":sum(x["onepass_changed_original_first_id"]
                                    for x in observations),
            "maximum_onpath_full_native_Linf_logit_shift":max(
                x["onepass_full_vocab_Linf_logit_shift"]
                for x in observations),
            "first_tokens_with_strict_margin_stability_bound":sum(
                x["margin_certifies_stability"] for x in observations),
            "real_decoder_calls_only_mechanical_comparison":real_decoder_forwards,
        },
        "decision":"EARLY_H320_SINGLE_PASS_ABI_MECHANIC_ONLY_NOT_QUALITY_OR_CPU_ACCEPTANCE",
    }
    out.mkdir(parents=True)
    (out/REPORT).write_text(json.dumps(
        result,sort_keys=True,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("S119Q1_ONEPASS_PHYSICAL_ORIGINAL_NATIVE "+json.dumps(
        result["summary"]|{"native_gate_gradient_norm":gate_grad_norm,
                           "accepted":False},sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser()
    for k in ("s93","side","s142","parent","parent_report",
              "s119n_report","out"):
        parser.add_argument("--"+k.replace("_","-"),required=True,type=Path)
    args=parser.parse_args()
    run(args.s93,args.side,args.s142,args.parent,args.parent_report,
        args.s119n_report,args.out)


if __name__=="__main__":
    main()
