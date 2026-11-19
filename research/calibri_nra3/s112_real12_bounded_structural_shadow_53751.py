"""S112 physical original CB59+GreenTunnel all12 source-only in-Calibri 53,751 shadow.

No GOLD or spent INTERNAL/dev or burned TaigaC downloaded/read. Requires
S109 original 616 J7 physics and S110 trained child physically SHA locked.
S112 53,751-weight research child is registered as a Calibri submodule,
all 53,743 old S110 FP16 bytes unchanged. New eight neural trust gates
initialized zero; old S105 BPE output exact at gate zero.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from research.calibri_nra3.s107_original12_source_only_shadow_53743 import Shadow12
from research.calibri_nra3.s103_in_calibri_morphosyntactic_affinity_37087 import NRA3RelationalMorphologyInCalibriWeights
from research.calibri_nra3.s112_in_calibri_bounded_structural_trust_53751 import (
    NRA3BoundedStructuralTrustInCalibriWeights,OLD,NEW,TOTAL,MAX_LOGIT_CORRECTION,
)

STAGE="CALIBRI_NRA3_S112_TRUE_ORIGINAL12_SOURCEONLY_BOUNDED_53751_NEURAL_SHADOW_GATE0"
SRC_SHA="5880ea08df06587a8a75a628dd4c55d94c89da9f1f2fa3aa6ab10c6ebd40f6b4"
OLD_CHILD="NRA3_S110_GSD_ORIGINAL12_NEW16656_STRUCTURAL_53743_IN_CALIBRI_FP16.safetensors"
SOURCE="NRA3_S109_GSD_TRAIN256_ALL12_ORIGINAL_H320_NO_GOLD.safetensors"
SOURCE_REPORT="NRA3_S109_REAL43MGT_GSD_TRAIN256_ALLVERBS_SOURCE_PROVENANCE.json"
OUT_CHILD="NRA3_S112_ORIGINAL12_BOUNDED_53751_IN_CALIBRI_ZERO_TRUST_FP16.safetensors"
REPORT="NRA3_S112_BOUNDED_ORIGINAL12_GSD_SOURCEONLY_NEURAL_ABI.json"
PREFIX="model.nra3_neural_weights."

def require(c,reason):
    if not bool(c):raise ValueError("NRA3 S112 true12 bounded neural shadow: "+reason)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

@torch.inference_mode()
def shadow(source12,source_manifest,s110_child,out):
    require(not out.exists(),"S112 only new physical shadow archive, no overwrite")
    require(source12.name==SOURCE and source_manifest.name==SOURCE_REPORT
            and s110_child.name==OLD_CHILD,
            "only genuine S109 original12 GSD source-only and S110 FP16 child")
    m=json.loads(source_manifest.read_text(encoding="utf8"))
    require(sha(source12)==SRC_SHA==m["ALL12_CPU_SAFETENSORS_SHA256"]
            and m["stage"]=="CALIBRI_NRA3_S109_REAL43MGT_ALL12_GSD_TRAIN256_ALLVERBS_SOURCE_ONLY_AFTER616"
            and m["S108_GSD_TRAIN_PINNED_SOURCE256_SHA256"]==
                 "4078f408f2355c55c29bca18504355379e2a42ef2144c9113dda9448b086de14"
            and m["old_616_J7_physically_replayed_BEFORE_NRA3_hooks"]
            and m["source_ONLY_no_S108_GSD_UD_GOLD_loaded_in_original_CPU"]
            and m["all_original_43M_GT_CB59_params_sha_pre_and_post_identical"]
            and m["NRA3_structural_neural_53743_weights_REGISTERED_INSIDE_actual_Calibri_model"]
            and m["original_S5_H8_port_exactly_matches_all12_H8_for_all_accepted"]
            and m["output_gate"]==0 and not m["candidate_accepted"],
            "S109 true original12 CB59+GT source and old 616 J7 protected")
    old_state=load_file(str(s110_child),device="cpu")
    require(sum(x.numel() for x in old_state.values())==OLD==53743
            and all(k.startswith(PREFIX) and v.dtype==torch.float16
                    and bool(torch.isfinite(v).all()) for k,v in old_state.items()),
            "S110 physically pretrained 53,743 FP16 in-Calibri child, no dev/report")
    n=NRA3BoundedStructuralTrustInCalibriWeights().eval()
    loaded=n.load_state_dict({k[len(PREFIX):]:v.float().clone()
                              for k,v in old_state.items()},strict=False)
    require(not loaded.unexpected_keys
            and loaded.missing_keys==["structural_trust_gate"]
            and torch.equal(n.structural_trust_gate,torch.zeros(8))
            and n.budget()["new_neural_parameters"]==TOTAL==53751
            and n.output_gate==0 and not n.candidate_accepted
            and not n.activation_allowed,
            "one new eight-value neural role trust registered, old S110 intact")
    holder=Shadow12()
    n.attach_to_RESEARCH_calibri(holder)
    owned={k:v for k,v in holder.state_dict().items() if k.startswith(PREFIX)}
    require(sum(p.numel() for p in owned.values())==TOTAL
            and set(old_state)<set(owned)
            and set(owned)-set(old_state)=={PREFIX+"structural_trust_gate"}
            and all(torch.equal(owned[k].half().cpu(),v) for k,v in old_state.items()),
            "S112 53,751 weights actually IN CALIBRI state_dict, 53,743 bit identical")
    prior=NRA3RelationalMorphologyInCalibriWeights().eval()
    prior.load_state_dict({k[len(PREFIX):]:v.float().clone()
                           for k,v in old_state.items()
                           if k[len(PREFIX):] not in (
                               "structural_predicate_query.weight",
                               "structural_argument_key.weight",
                               "structural_role_bilinear",
                               "structural_native_BPE_distance.weight",
                               "structural_competition_gate")},
                          strict=True)
    data=load_file(str(source12),device="cpu")
    h=data["original_Calibri43M_GT_ALL12_native_H320_FP16_READONLY"]
    valid=data["native_BPE_source_valid_mask"]
    require(h.shape[1]==12 and h.shape[-1]==320 and h.dtype==torch.float16
            and h.shape[0]==m["n_accepted_source_only_BPE_rows"]==249
            and int(valid.sum())==m["n_genuine_original_native_BPE_tokens"]==28223,
            "genuine original12 GSD BPE from frozen original CB59 GreenTunnel")
    results=[]
    for ix in (0,128):
        inp=h[ix:ix+1].float().detach()
        v=valid[ix:ix+1]
        a=prior(inp,v)
        b=n(inp,v)
        keys=("event_mention_logits","role_arc_logits","event_state_logits",
              "mini_area_logits","layer_router_weights","memory_slot_attention",
              "upos_logits","case_logits","number_logits","voice_logits")
        require(all(torch.equal(getattr(a,k),getattr(b,k)) for k in keys),
                "new structural trust0 MUST preserve ALL original real source logits exactly")
        # Synthetic high-trust ablation ON ORIGINAL SOURCE ONLY, no corpus annotation.
        # No weight updates persisted; trust gate restored to zero.
        n.structural_trust_gate.fill_(3.)
        alt=n(inp,v)
        pair=v[:,None,:,None]&v[:,None,None,:]
        delta=(alt.role_arc_logits-a.role_arc_logits).abs()
        require(not bool((delta[pair.expand_as(delta)]>MAX_LOGIT_CORRECTION+1.e-5).any()),
                "source-only high trust MUST not exceed one logit per valid candidate")
        n.structural_trust_gate.zero_()
        c=n(inp,v)
        require(torch.equal(a.role_arc_logits,c.role_arc_logits),
                "zero new 8 trust gates restored exact prior after finite ablation")
        results.append({"original_GSD_source_row":ix,
                        "n_original_native_BPE":int(v.sum()),
                        "all_10_original_neural_tensors_BIT_IDENTICAL_at_TRUST0":True,
                        "max_source_role_logit_correction_at_trust3":
                             float(delta[pair.expand_as(delta)].max())})
    weights={PREFIX+k:v.detach().half().cpu().contiguous()
             for k,v in n.state_dict().items()}
    require(sum(z.numel() for z in weights.values())==53751
            and all(torch.equal(weights[k],v) for k,v in old_state.items())
            and torch.count_nonzero(weights[PREFIX+"structural_trust_gate"])==0,
            "53,743 previous FP16 + 8 only new zero-gated FP16 trust param exact")
    out.mkdir(parents=True,exist_ok=False)
    save_file(weights,str(out/OUT_CHILD))
    back=load_file(str(out/OUT_CHILD),device="cpu")
    require(set(back)==set(weights)
            and all(torch.equal(back[k],v) for k,v in weights.items()),
            "S112 in-model FP16 zero-gate checkpoint roundtrip")
    report={
        "stage":STAGE,
        "old616_J7_and_CB59_GT_physical_source_SHA":SRC_SHA,
        "prior_S110_53743_weights_SHA256":sha(s110_child),
        "new_S112_53751_8trust_in_calibri_FP16_SHA256":sha(out/OUT_CHILD),
        "original_37k_NRA2_mini_areas_memory_morphology_router_unchanged":True,
        "old_53743_learnt_neural_weights_FP16_exact":True,
        "eight_new_neural_role_trust_params_in_calibri":8,
        "prior_full_neural_params":OLD,
        "new_full_neural_params":TOTAL,
        "maximum_individual_native_BPE_logit_correction":MAX_LOGIT_CORRECTION,
        "zero_trust_restores_original_old_all_neural_role_logits":True,
        "old_calibri_GT_616_original_source_immutable":True,
        "GSD_SOURCE_ONLY_NO_UD_gold_or_teacher_runtime":True,
        "S110_internal_dev_SPENT_not_read":True,
        "no_old_dev_or_burned_TaigaC_used":True,
        "physical_original12_source_only_ablations":results,
        "new_neural_8_role_trust_params_UNTRAINED":True,
        "neural_quality_OR_independent_transfer_NOT_PROVEN":True,
        "no_active_j7_online_causal_or_allin_bpp_CPU_certification":True,
        "CC_BY_SA_4_legal_review_before_commercial":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    (out/REPORT).write_text(json.dumps(report,ensure_ascii=False,
                         sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("NRA3_S112_REAL_ORIGINAL12_53751_BOUNDED_NEURAL_SOURCE_ONLY",
          json.dumps({"source_SHA":SRC_SHA,"n_full":TOTAL,"extra_trust":NEW,
                      "checks":results,"gate":0},sort_keys=True),flush=True)
    return report

def main():
    a=argparse.ArgumentParser()
    for k in ("source12","source_manifest","s110_child","out"):
        a.add_argument("--"+k.replace("_","-"),type=Path,required=True)
    args=a.parse_args()
    shadow(args.source12,args.source_manifest,args.s110_child,args.out)

if __name__=="__main__":
    main()
