"""S107 physical SOURCE-ONLY Calibri CB59 GT true12 shadow ABI, zero gold.

Run strictly after successful original S104 protected616 CPU + S105.
Only archived frozen original12 source-only H320, new S107 child and the
existing S105 in-Calibri neural FP16 weights are examined. NEVER import
S105 internal diagnostic results or independent Taiga TRAIN-c.
NO neural training, NO new output activation, NO original model mutation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from safetensors.torch import load_file,save_file
from research.calibri_nra3.s103_in_calibri_morphosyntactic_affinity_37087 import (
    NRA3RelationalMorphologyInCalibriWeights,
)
from research.calibri_nra3.s107_in_calibri_structural_role_competition_53743 import (
    NRA3StructuralCompetitionInCalibriWeights,TOTAL_NEURAL_PARAMETERS,
    OLD_NEURAL_PARAMETERS,NEW_KEYS,
)

STAGE="CALIBRI_NRA3_S107_PHYSICAL_ORIGINAL12_S104_SOURCE_ONLY_53743_SHADOW_AFTER616"
REAL_SOURCE_SHA="c46e236072a6b39eb7a0e02fdc20c622c8b391785b7c2b21b99a416b8d902a2e"
S104_SOURCE="NRA3_S104_TAIGA_TRAINE256_ALL12_ORIGINAL_H320_NO_GOLD.safetensors"
S104_REPORT="NRA3_S104_REAL43MGT_TAIGA_TRAINE256_ALLVERBS_SOURCE_PROVENANCE.json"
OLD_S105_CHILD="NRA3_S105_TAIGA_E_STRUCTURAL_AFFINITY_37087_IN_CALIBRI_FP16.safetensors"
NEW_S107_CHILD="NRA3_S107_ORIGINAL12_53743_IN_CALIBRI_NEURAL_UNTRAINED_GATE0_FP16.safetensors"
REPORT="NRA3_S107_ORIGINAL12_SOURCE_ONLY_NEURAL_53743_IDENTITY_MANIFEST.json"
PREFIX="model.nra3_neural_weights."

def require(c,msg):
    if not bool(c):raise ValueError("NRA3 S107 true-original12 source-only research: "+msg)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Shadow12(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.layers=nn.ModuleList([nn.Identity() for _ in range(12)])


@torch.inference_mode()
def shadow(source_path,source_report,old_child_file,out):
    require(not out.exists(),"never overwrite physical S107 evidence")
    require(source_path.name==S104_SOURCE and source_report.name==S104_REPORT
            and old_child_file.name==OLD_S105_CHILD,
            "only exact S104 source and immutable S105 neural child allowed")
    oldmeta=json.loads(source_report.read_text(encoding="utf8"))
    require(sha(source_path)==REAL_SOURCE_SHA==
             oldmeta["ALL12_CPU_SAFETENSORS_SHA256"]
            and oldmeta["old_616_J7_physically_replayed_BEFORE_NRA3_hooks"]
            and oldmeta["source_ONLY_no_S103_ALL_VERBS_UD_morph_multiedge_gold_loaded_in_original_CPU"]
            and oldmeta["NRA3_morph_neural_37087_weights_REGISTERED_INSIDE_actual_Calibri_model"]
            and oldmeta["n_accepted_source_only_BPE_rows"]==256
            and oldmeta["n_genuine_original_native_BPE_tokens"]==18096
            and oldmeta["original_S5_H8_port_exactly_matches_all12_H8_for_all_accepted"]
            and oldmeta["output_gate"]==0 and not oldmeta["candidate_accepted"],
            "physical original12+GT SHA + old616 J7 before S107 neural attachment")
    oldweights=load_file(str(old_child_file),device="cpu")
    require(all(k.startswith(PREFIX) for k in oldweights)
            and sum(x.numel() for x in oldweights.values())==OLD_NEURAL_PARAMETERS
            and all(x.dtype==torch.float16 for x in oldweights.values()),
            "S105 genuine 37,087 in-model neural child FP16, no old diagnostics")
    torch.manual_seed(1071119)
    original=NRA3RelationalMorphologyInCalibriWeights().eval()
    original.load_state_dict({k[len(PREFIX):]:v.float().clone()
                              for k,v in oldweights.items()},strict=True)
    n=NRA3StructuralCompetitionInCalibriWeights().eval()
    result=n.load_state_dict({k[len(PREFIX):]:v.float().clone()
                               for k,v in oldweights.items()},strict=False)
    require(not result.unexpected_keys
            and set(result.missing_keys)==NEW_KEYS
            and n.budget()["new_neural_parameters"]==TOTAL_NEURAL_PARAMETERS
            and torch.equal(n.structural_competition_gate,torch.zeros(8)),
            "all existing 37087 weights loaded without loss + new 16656 zero-gated")
    owned=Shadow12()
    n.attach_to_RESEARCH_calibri(owned)
    complete={k:v for k,v in owned.state_dict().items()
              if k.startswith(PREFIX)}
    require(sum(p.numel() for p in complete.values())==TOTAL_NEURAL_PARAMETERS
            and set(oldweights)<set(complete)
            and all(torch.equal(complete[k].half().cpu(),v)
                    for k,v in oldweights.items()),
            "S107 physically IN Calibri state_dict and prior FP16 weights bit-identical")
    source=load_file(str(source_path),device="cpu")
    h=source["original_Calibri43M_GT_ALL12_native_H320_FP16_READONLY"]
    valid=source["native_BPE_source_valid_mask"]
    require(h.shape[0]==256 and h.shape[1]==12 and h.shape[-1]==320
            and h.dtype==torch.float16 and valid.dtype==torch.bool
            and valid.shape==(256,h.shape[2]) and int(valid.sum())==18096,
            "original full source 12x320 actual native BPE without gold")
    exact=[]
    # Fixed deterministic first and middle source-only samples; NEVER
    # choose rows based on labels, predictor quality or diagnostic scores.
    for idx in (0,128):
        x=h[idx:idx+1].float().detach()
        mask=valid[idx:idx+1]
        a=original(x,mask)
        b=n(x,mask)
        names=("event_mention_logits","role_arc_logits",
               "event_state_logits","mini_area_logits",
               "layer_router_weights","memory_slot_attention",
               "upos_logits","case_logits","number_logits","voice_logits")
        require(all(torch.equal(getattr(a,k),getattr(b,k)) for k in names),
                "true original all12 source-only S105 logits EXACT at zero new structural gate")
        exact.append({"source_original_row":idx,"input_original_native_BPE":int(mask.sum()),
                      "all10_neural_logits_BIT_IDENTICAL":True})
    weights={PREFIX+k:v.detach().half().cpu().contiguous()
             for k,v in n.state_dict().items()}
    require(sum(v.numel() for v in weights.values())==TOTAL_NEURAL_PARAMETERS
            and all(torch.equal(weights[k],v) for k,v in oldweights.items()),
            "physical S107 FP16 inherited full child unchanged")
    out.mkdir(parents=True,exist_ok=False)
    save_file(weights,str(out/NEW_S107_CHILD))
    reread=load_file(str(out/NEW_S107_CHILD),device="cpu")
    require(set(reread)==set(weights)
            and all(torch.equal(reread[k],v) for k,v in weights.items()),
            "save exact new 53743 S107 child in-model FP16 checkpoint")
    m={
        "stage":STAGE,
        "original_CB59_GT_ALL12_source_SHA":REAL_SOURCE_SHA,
        "old_616_J7_physically_replayed_by_S104":True,
        "full_original_native_BPE_source_n":18096,
        "previous_S105_37087_neural_FP16_SHA":sha(old_child_file),
        "new_S107_53743_in_Calibri_neural_FP16_SHA":sha(out/NEW_S107_CHILD),
        "new_structural_params":16656,
        "old_S105_in_Calibri_params":OLD_NEURAL_PARAMETERS,
        "new_in_Calibri_neural_params":TOTAL_NEURAL_PARAMETERS,
        "original_model_and_GreenTunnel_NOT_retrained_or_modified":True,
        "eight_original_NRA2_mini_areas_preserved":True,
        "original_all12_learned_layer_router_preserved":True,
        "original_16_in_weight_memory_slots_preserved":True,
        "original_S105_morphology_heads_and_morph_affinities_preserved":True,
        "neural_gold_free_full12_original_native_BPE_tests":exact,
        "no_original_Taiga_DEV_TEST_or_reserved_TRAIN_c_access":True,
        "old_S96_S99_S102_S105_internal_devs_SPENT_not_accessed":True,
        "no_new_NRA3_training_or_quality_or_factuality_claim":True,
        "no_online_causal_or_full_model_bpp_le07_CPU_ge90_proof":True,
        "third_party_CC_BY_SA_4_attribution_and_legal_review_before_commercial":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    (out/REPORT).write_text(
        json.dumps(m,indent=2,ensure_ascii=False,sort_keys=True)+"\n",
        encoding="utf8")
    print("NRA3_S107_REAL_ORIGINAL12_53743_IN_CALIBRI_ZERO_GATE_SUCCESS",
          json.dumps({
            "old616_preserved":True,
            "original_BPE_source":18096,
            "old_neural_weights_exact":OLD_NEURAL_PARAMETERS,
            "new_structural_params":16656,
            "all_neural_child_params":TOTAL_NEURAL_PARAMETERS,
            "source_only_bit_identical_rows":[0,128],
            "new_child_SHA":m["new_S107_53743_in_Calibri_neural_FP16_SHA"],
            "output_gate":0,
          },sort_keys=True),flush=True)
    return m

def main():
    a=argparse.ArgumentParser()
    for key in ("source","source_report","previous_s105","out"):
        a.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    x=a.parse_args()
    shadow(x.source,x.source_report,x.previous_s105,x.out)

if __name__=="__main__":
    main()
