"""Real universal Green Tunnel operator coverage from AUTHENTIC paid 135M v186.

No teacher, no training and no QA here; this proves a dynamic
all-211-matrix schedule is compatible with the actual model, not just
a synthetic toy. It is a DIFFERENT outcome from model quality.
"""
import argparse
import json
from pathlib import Path
from hashlib import sha256

from direct_codebook_v59 import CB59
from green_tunnel.contract import PhysicalSource,verify_physical_source
from green_tunnel.adapter_cb59 import plan_complete_CB59_recovery


REAL_135M_SHA="933daaecf9011f77e3705308faab9d6074b5fb7c4c847d5e1bdb830ad47b965a"
ORIGINAL_UNIQUE_135M=134_515_008
REAL_135M_BYTES=11_229_303


def run(source,report):
    blob=Path(source).read_bytes()
    original=CB59(blob)
    meta=original.meta
    x=PhysicalSource(
        "CB59 physical original 135M",
        original_unique_parameters=ORIGINAL_UNIQUE_135M,
        physical_whole_bytes=REAL_135M_BYTES,
        physical_sha256=REAL_135M_SHA,
        aliases=tuple(sorted(meta["source_alias_keys"].items())))
    verified=verify_physical_source(
        x,blob,decode=CB59,
        verify_alias=lambda archive,ties:(
            archive.meta["original_unique_parameter_count"]==ORIGINAL_UNIQUE_135M and
            tuple(sorted(archive.meta["source_alias_keys"].items()))==ties and
            archive.handles["model.embed_tokens.weight"] is archive.handles["lm_head.weight"]))
    plan=plan_complete_CB59_recovery(meta,2)
    if (plan["steps_per_cycle"]!=211 or
        len(plan["precommitted_targets"])!=2 or
        len({name for sched in plan["precommitted_targets"] for name in sched})!=211):
        raise AssertionError("Real 135M source not all covered by Green Tunnel")
    result={"source":verified,
        "actual_135M_pretrained_source_SHA256":REAL_135M_SHA,
        "actual_real_211_unique_paid_2d_operators":211,
        "model_architecture_specific_211_inputs_from_physical_metadata":True,
        "all_211_given_discrete_update_turn_per_recovery_cycle":True,
        "planning_cycles":2,
        "planned_422_discrete_source_operator_turns":True,
        "all_original_135M_shared_aliases_preserved":True,
        "planning_only_NO_MODEL_TRAINING_DONE_HERE":True,
        "no_claim_fp32_ANSWER_PARITY":True,
        "no_360M_or_larger_models_evaluated":True,
        "plan":plan}
    output=Path(report)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print(json.dumps({"stage":"v197_real_135M_all_matrix_source_contract_PASS",
        "original_full_bytes":len(blob),
        "all_unique_2d_matrices":plan["steps_per_cycle"],
        "complete_0p700_bpp":x.actual_bpp,
        "all_layer_depths_scheduled":True,
        "original_trained_model_quality_pass":False,
        "training_not_run_in_this_job":True}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("physical_v186_135M_file",type=Path)
    p.add_argument("output_JSON",type=Path)
    args=p.parse_args()
    run(args.physical_v186_135M_file,args.output_JSON)
