"""S193 compile-only audit of atomic 584-protected J7 training transaction.

Checks strict prior S192 manifest, physical all-in source size/bpp.
No real model training or CPU J7 rerun here; successful S193 means the
transaction's mock preflight is safe, NOT that any new weights were trained.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import sys

from calibri2.s182.real_cpu_j7_shadow_guard import DENOM

S193_SOURCES=(
    "research/calibri2/s193/atomic_one_step_584.py",
    "research/calibri2/s193/test_atomic_one_step_584.py",
    "research/calibri2/s193/atomic_584_preflight_report.py",
)


def prepare(manifest, files):
    if (manifest.get("stage")!="S192_PROTECTED_NEW_TRAIN_32_READINESS_ONLY"
        or manifest.get("full_EOS_and_first_token_protected_584")!=584
        or manifest.get("new_TRAIN_paraphrase_32_count")!=32
        or manifest.get("source_S190_real_CPU_new32_full_EOS_correct")!=0
        or manifest.get("new_32_not_in_prior_584") is not True
        or manifest.get("pinned_parent_real_CPU_456_128") is not True
        or manifest.get("training_executed") is not False
        or manifest.get("new_weights_emitted") is not False
        or manifest.get("model_accepted") is not False):
        raise AssertionError("S193 immutable S192 original+new TRAIN readiness missing")
    c=manifest.get("training_contract",{})
    if (not c.get("all_456_plus_128_TRUE_CPU_full_EOS_and_first_strict_parity")
        or not c.get("atomic_weights_and_Adam_rollback_on_any_failed_gate")
        or not c.get("candidate_physical_FP16_save_reload_required")
        or c.get("bounded_transaction_optimizer_steps_max")!=8
        or c.get("all_in_physical_bpp_max")!=.7
        or c.get("paired_same_CPU_speed_ratio_min")!=.9
        or c.get("LIVE_Jessica_Render_Supabase_APK_untouched") is not True):
        raise AssertionError("S193 S192 J7/rollback/bpp/CPU contract missing")
    if not files or any(not p.exists() or not p.is_file() for p in files):
        raise AssertionError("S193 training contract code file missing")
    source_charge=sum(p.stat().st_size for p in files)
    bpp=manifest["pretraining_max_bpp_including_research_code"]+8*source_charge/DENOM
    if bpp>.7:
        raise AssertionError("S193 physical all-in source bpp exceeds 0.7")
    return {
        "stage":"S193_ONE_STEP_GUARD_PREFLIGHT_ONLY",
        "parent_S192_run":43342581672,
        "parent_S187_456_and_S122_128_real_CPU_validated_in_prior_run":True,
        "protected_full_and_first_token_train_identities":584,
        "new_TRAIN_32_probes":32,
        "new_TRAIN_full_EOS_baseline":0,
        "actual_optimizer_transactions_performed_in_S193":0,
        "actual_43M_GPU_weights_trained_in_S193":False,
        "actual_real_J7_456_128_32_rerun_in_S193":False,
        "S193_atomic_step_synthetic_6_unittests_required":True,
        "real_J7_callback_mandatory_before_any_future_commit":True,
        "S125_hardcoded_353_guard_rejected_as_584_incompatible":True,
        "physical_FP16_persist_reload_and_protected_output_digest_mandatory":True,
        "weights_and_Adam_exact_rollback_mandatory":True,
        "new32_positive_full_EOS_gain_mandatory":True,
        "no_inference_time_gold_SHA_or_QA_TEST":True,
        "physical_bpp_budget_charging_S193_sources":bpp,
        "remaining_bpp_headroom":.7-bpp,
        "model_accepted":False,
        "LIVE_Jessica_Render_Supabase_APK_untouched":True,
        "code_sha256":hashlib.sha256(
            b"".join(p.read_bytes() for p in files)).hexdigest(),
    }


def main():
    if len(sys.argv)!=3:
        raise SystemExit("S193 requires pinned S192 JSON and output JSON")
    src,out=(Path(x) for x in sys.argv[1:])
    if out.exists():
        raise FileExistsError("S193 immutable preflight report cannot overwrite")
    manifest=json.loads(src.read_text(encoding="utf-8"))
    result=prepare(manifest,[Path(p) for p in S193_SOURCES])
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,sort_keys=True,indent=2),encoding="utf-8")
    print(json.dumps(result,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
