"""S192: strict PRE-TRAIN readiness for protected 32 new TRAIN paraphrases.

Prepare only deterministic curriculum and immutable guard contract. DOES NOT
modify ANY weights, run training, accept a model, or perform Jessica chat tests.
S190's 32 cases are now authorized TRAIN (not evidence of generalization).
Original 456 + S122 128 = 584 protected real full+EOS and first-token anchors.
Any future model update requires a separate explicit bounded J7 transaction,
physical FP16 reload and whole 584 TRUE CPU parity with atomic rollback.
No QA/TEST/heldout sources or inference-time SHA/labels/answer lookup.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from calibri2.s190.real_cpu_new_train_paraphrase_probe import new_train_paraphrase_probes
from calibri2.s182.real_cpu_j7_shadow_guard import DENOM

ORIGINAL_COUNT=456
S122_COUNT=128
NEW_TRAIN_COUNT=32
PHYSICAL_BPP_MAX=0.7
MIN_CPU_RATIO=0.9
SUPPORTED_GROUPS={
    "owner_A_paraphrase":8,
    "owner_B_paraphrase":8,
    "reordered_owner_paraphrase":8,
    "numeric_new_wording":8,
}
CHARGE_S192_SOURCE=(
    "research/calibri2/s191/real_new_train_error_attribution.py",
    "research/calibri2/s191/test_real_new_train_error_attribution.py",
    "research/calibri2/s192/prepare_protected_new_train.py",
    "research/calibri2/s192/test_prepare_protected_new_train.py",
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_source_readiness(
    tok, original_sha, s122_sha, s190, s191, s122_digest: str,
    *, charge_bytes: int,
) -> dict:
    """Return manifest with SHA-only samples, not answers or gradients.

    A tokenizer must be the immutable original model's exact tokenizer.
    Original and S122 hashes must be independently reconstructed from
    authorized TRAIN sources, NEVER from inspected QA/TEST.
    """
    if (len(original_sha)!=ORIGINAL_COUNT or len(s122_sha)!=S122_COUNT
            or not all(isinstance(s,str) and len(s)==64 for s in original_sha | s122_sha)
            or original_sha & s122_sha):
        raise AssertionError("S192 original 456 and S122 128 must be unique/disjoint")
    if not isinstance(s122_digest,str) or len(s122_digest)!=64:
        raise AssertionError("S192 pinned S122 curriculum digest required")
    actual_s122_digest=digest("\n".join(sorted(s122_sha)))
    if actual_s122_digest!=s122_digest:
        raise AssertionError("S192 reconstructed S122 original 128 TRAIN digest mismatch")
    if (s190.get("event")!="S190_REAL_CPU_32_UNSEEN_NEW_TRAIN_PARAPHRASES_STRICT_SOURCE_ABSTAIN"
            or s190.get("original_real_CPU_FULL_EOS_456")!=456
            or s190.get("new128_real_CPU_FULL_EOS")!=128
            or s190.get("original_first_full_forgetting")!=0
            or s190.get("real_CPU_full_J7_new_TRAIN_all32") is not True
            or s190.get("no_gold_or_SHA_at_inference_or_routing") is not True
            or s190.get("no_neural_weights_updated") is not True
            or s190.get("model_accepted") is not False
            or s190.get("all_in_physical_bpp_charging_S188_S189_S190_source",1)>.7
            or s190.get("original_CPU_S101_ratio",0)<MIN_CPU_RATIO):
        raise AssertionError("S192 pinned real S190 584 old/new CPU J7 provenance failed")
    if (s191.get("stage")!="S191_POSTHOC_REAL_S190_NEW_TRAIN_FAILURE_MODES_NO_NEURAL"
            or s191.get("S190_immutable_run")!=43340296813
            or s191.get("new_train_probe_count")!=32
            or s191.get("new_train_full_EOS_correct")!=s190.get("NEW_TRAIN_full_EOS_correct")
            or s191.get("new_train_first_correct")!=s190.get("NEW_TRAIN_first_correct")
            or s191.get("not_a_new_neural_generalization_result") is not True
            or s191.get("model_accepted") is not False):
        raise AssertionError("S192 immutable S191 SHA-only failure attribution mismatched")
    probes=new_train_paraphrase_probes()
    groups=Counter()
    new_hashes=set()
    source_groups={}
    expected_length=Counter()
    for question,answer,group in probes:
        if group not in SUPPORTED_GROUPS:
            raise AssertionError("S192 new TRAIN family not predeclared")
        sha=digest(question)
        if sha in new_hashes or sha in original_sha or sha in s122_sha:
            raise AssertionError("S192 new TRAIN duplicate or protected 584 collision")
        new_hashes.add(sha)
        source_groups[sha]=group
        groups[group]+=1
        expected=tok.encode(answer, add_special_tokens=False)
        need=1 if group=="numeric_new_wording" else 2
        if (len(expected)!=need or any(type(t) is not int or t<0
                                      or t==tok.eos_token_id for t in expected)):
            raise AssertionError("S192 new TRAIN target token + EOS ABI incompatible")
        expected_length[len(expected)+1]+=1
    if dict(groups)!=SUPPORTED_GROUPS or len(new_hashes)!=NEW_TRAIN_COUNT:
        raise AssertionError("S192 exact NEW TRAIN 8x4 fixed set changed")
    audit=s190.get("hashed_per_prompt_audit",[])
    if len(audit)!=32 or {r.get("prompt_sha256") for r in audit}!=new_hashes:
        raise AssertionError("S192 actual S190 real frozen CPU probe source SHA drift")
    if sum(r.get("full_EOS_correct") is True for r in audit)!=s190.get("NEW_TRAIN_full_EOS_correct"):
        raise AssertionError("S192 actual S190 output/count mismatched")
    if any(source_groups.get(r.get("prompt_sha256")) != r.get("group") for r in audit):
        raise AssertionError("S192 original new TRAIN SHA-to-source-family drift")
    if not isinstance(charge_bytes,int) or charge_bytes<0:
        raise AssertionError("S192 exact nonnegative runtime code size required")
    inherited_bpp=s190["all_in_physical_bpp_charging_S188_S189_S190_source"]
    reserved_bpp=inherited_bpp+(8*charge_bytes/DENOM)
    if reserved_bpp>PHYSICAL_BPP_MAX:
        raise AssertionError("S192 physical bpp headroom insufficient before training")
    return {
        "stage":"S192_PROTECTED_NEW_TRAIN_32_READINESS_ONLY",
        "original_456_SHA_count":456,
        "S122_protected_128_SHA_count":128,
        "full_EOS_and_first_token_protected_584":584,
        "new_TRAIN_paraphrase_32_count":32,
        "NEW_TRAIN_per_family":dict(sorted(groups.items())),
        "NEW_TRAIN_target_lengths_including_EOS":dict(sorted(expected_length.items())),
        "original_S122_TRAIN_digest":s122_digest,
        "new_TRAIN_32_hashset_digest":digest("\n".join(sorted(new_hashes))),
        "new_32_not_in_prior_584":True,
        "source_S190_real_CPU_new32_full_EOS_correct":s190["NEW_TRAIN_full_EOS_correct"],
        "source_S190_real_CPU_new32_first_correct":s190["NEW_TRAIN_first_correct"],
        "source_S191_failures":s191["failure_modes"],
        "protected_parent_S187_run":43336858062,
        "protected_independent_S188_run":43338233327,
        "protected_new32_real_S190_run":43340296813,
        "protected_S191_posthoc_run":43341738229,
        "pinned_parent_real_CPU_456_128":True,
        "baseline_bpp":inherited_bpp,
        "pretraining_max_bpp_including_research_code":reserved_bpp,
        "remaining_bpp_headroom":PHYSICAL_BPP_MAX-reserved_bpp,
        "training_contract":{
            "isolated_branch_only":True,
            "checkpoint_source":"frozen S142 FP16 core + S160 J7 slot/GT + S187 protected selector",
            "new32_all_are_TRAIN_not_independent_validation":True,
            "training_target":"frozen tokenizer NEW TRAIN full content tokens and EOS",
            "trainable_scope":"existing additive Green Tunnel / new channels only; freeze parent weights",
            "bounded_transaction_optimizer_steps_max":8,
            "old_loss_gradient_projection":"S131 first-order only, never evidence of retention",
            "after_Adam_displacement_projection":"S142 first-order only, never evidence of retention",
            "candidate_physical_FP16_save_reload_required":True,
            "all_456_plus_128_TRUE_CPU_full_EOS_and_first_strict_parity":True,
            "new32_full_EOS_positive_gain_required_after_real_CPU":True,
            "all_in_physical_bpp_max":PHYSICAL_BPP_MAX,
            "paired_same_CPU_speed_ratio_min":MIN_CPU_RATIO,
            "atomic_weights_and_Adam_rollback_on_any_failed_gate":True,
            "no_inference_lookup_by_SHA_or_reference_answers":True,
            "QA_TEST_heldout_disallowed_for_training_routing_or_selection":True,
            "LIVE_Jessica_Render_Supabase_APK_untouched":True,
        },
        "training_executed":False,
        "new_weights_emitted":False,
        "model_accepted":False,
        "no_QA_TEST_or_Jessica_chat":True,
    }


def main() -> None:
    if len(sys.argv)!=6:
        raise SystemExit("S192 requires pinned S179,S122,S190,S191 JSON sources and output JSON")
    s179_file,s122_file,s190_file,s191_file,out_file=map(Path,sys.argv[1:])
    if out_file.exists():
        raise FileExistsError("S192 immutable readiness manifest cannot overwrite")
    original=json.loads(s179_file.read_text())
    source122=json.loads(s122_file.read_text())
    s190=json.loads(s190_file.read_text())
    s191=json.loads(s191_file.read_text())
    if original.get("stage")!="S179_TRACE_ONLY" or original.get("labels_used") is not False:
        raise AssertionError("S192 S179 no-gold original CPU trace required")
    src456={r["prompt_sha256"] for r in original["cases"]}
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    from calibri2.s122.independent_binding_train_curriculum import materialize_train_only
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    from calibri2.s30 import matched_dense_fp32_train as s30
    s30.TOK=tok  # exact original S122 tokenizer ABI
    new128,checksum,_=materialize_train_only(tok)
    sha128={digest(x["question"]) for x in new128}
    if checksum!=source122.get("new_TRAIN_question_hashset_sha256"):
        raise AssertionError("S192 immutable S122 128 curriculum manifest digest mismatch")
    charge=sum(Path(path).stat().st_size for path in CHARGE_S192_SOURCE)
    report=verify_source_readiness(tok,src456,sha128,s190,s191,checksum,charge_bytes=charge)
    out_file.parent.mkdir(parents=True,exist_ok=True)
    out_file.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
