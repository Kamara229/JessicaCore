#!/usr/bin/env python3
"""S171: frozen S170 artifact-only diagnosis. No runtime/model/test rerun."""
import json
import sys
from pathlib import Path

from calibri2.s169.owner_segment_offline import owner_value

def main():
    inp,out=map(Path,sys.argv[1:])
    source=json.loads((inp/"S168_NOVEL_TRAIN_FRAMES_64_GENERALIZATION.json").read_text())
    guarded=json.loads((inp/"S170_INTEGRATED_STRUCTURAL_OWNER_FINAL_GATE.json").read_text())
    assert len(source["cases"])==64
    assert source["candidate_full_correct"]==63, "Unexpected S170 baseline; re-evaluate"
    assert guarded["safe_in_research"] is False and guarded["model_accepted"] is False
    misses=[row for row in source["cases"] if not row["candidate_correct"]]
    assert len(misses)==1
    detail=[]
    for row in misses:
        question=row["question"]
        selection=owner_value(question)  # inference from question ONLY
        actual=row["answer"]  # label used for analysis ONLY
        detail.append({
            "id":row["id"],"question":question,
            "source_only_parser":selection,
            "parser_matches_expected_after_scoring":selection==actual,
            "expected_phrase_for_posthoc_diagnostic_only":actual,
            "original_S142_correct":row["parent_correct"],
            "S170_selected_full_EOS_ids":row["candidate_FULL_EOS_ids"],
            "gold_full_EOS_ids_posthoc_only":row["expected_FULL_EOS_ids"],
            "fastpath":row["fastpath"],
            "gate_reason_counters":row["gate_counter_deltas"],
            "S93_route":row["S93_route"]
        })
    diagnosis={
        "stage":"S171_FROZEN_S170_ONE_MISS_DIAGNOSIS",
        "source_S170_run":43293552917,
        "S170_novel_correct":source["candidate_full_correct"],
        "S170_J7_safe":guarded["protected_J7_safe"],
        "S170_full_acceptance":guarded["safe_in_research"],
        "fails":detail,
        "no_gold_at_inference":True,"QA_TEST_never_used":True,
        "no_neural_weights_updated":True,"LIVE_untouched":True,
        "model_accepted":False
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"S171_S170_ONE_MISS_DIAGNOSTIC.json").write_text(
        json.dumps(diagnosis,ensure_ascii=False,indent=2,sort_keys=True))
    print(json.dumps({"event":"S171_FROZEN_S170_MISS",
        "miss_id":detail[0]["id"],
        "question":detail[0]["question"],
        "source_only_parser":detail[0]["source_only_parser"],
        "parser_matches_expected":detail[0]["parser_matches_expected_after_scoring"],
        "original_S142_correct":detail[0]["original_S142_correct"],
        "fastpath":detail[0]["fastpath"],
        "gate_reason_counters":detail[0]["gate_reason_counters"],
        "candidate_full":detail[0]["S170_selected_full_EOS_ids"],
        "expected_full":detail[0]["gold_full_EOS_ids_posthoc_only"],
        "model_accepted":False},sort_keys=True),flush=True)
if __name__=="__main__":
    main()
