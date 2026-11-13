"""S194: bind actual S190 CPU J7 decoder to S193 584+32 transaction gate.

This is a FROZEN BASELINE proof, NOT a trained candidate. S194 executes
exactly the real S190 CPU J7 on 456+128+32, intercepts actual generated
token IDs BEFORE reading any answers, then reconstructs S193's typed
584+32 certificate AFTER all 616 predictions complete.

Future S195 step must: persist its changed FP16, use a SEPARATE process
to run this exact decoder with candidate sidecar, measure paired CPU speed,
and pass its real certificate to S193.one_protected_step. Candidate
checkpoints cannot claim baseline speed 1.0; fail closed until measured.
Original 456 S87 answers and S122 answers are used for POSTHOC SCORING,
never in source selection or neural routing. No QA/TEST, LIVE or merge.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from calibri2.s193.atomic_one_step_584 import J7Retention584

REAL_CAPTURE={}

def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def output_digest(preds: dict[str,list[int]]) -> str:
    if not preds or any(not isinstance(k,str) or len(k)!=64 or
                        not isinstance(v,list) or
                        any(type(i) is not int or i<0 for i in v)
                        for k,v in preds.items()):
        raise AssertionError("S194 full generated CPU J7 ids required")
    canonical=json.dumps([[k,preds[k]] for k in sorted(preds)],
                         separators=(",",":"),ensure_ascii=True)
    return sha_text(canonical)

def make_real_584_certificate(
    *, old_pred, old_target, new_pred, new_target, probe_pred, probe_target,
    eos_id: int, source_bpp: float, speed_ratio: float,
    checkpoint_sha256: str, physically_reloaded: bool,
    frozen_parent: bool, captured_before_gold: bool,
) -> J7Retention584:
    """Post-inference only. New32 must come from actual CPU J7 tokens."""
    if not captured_before_gold or type(eos_id) is not int or eos_id<0:
        raise AssertionError("S194 gold-free real CPU inference capture not confirmed")
    for p,t,count in ((old_pred,old_target,456),
                      (new_pred,new_target,128),
                      (probe_pred,probe_target,32)):
        if len(p)!=count or len(t)!=count or set(p)!=set(t):
            raise AssertionError("S194 exact 456+128+32 source SHA alignment required")
        if any(not isinstance(x,list) or not x or x[-1]!=eos_id
               for x in t.values()):
            raise AssertionError("S194 target ABI requires true EOS on all source TRAIN")
        # Verify generated id lists are real numeric token sequences.
        output_digest(p)
    old_full=sum(old_pred[k]==old_target[k] for k in old_pred)
    new_full=sum(new_pred[k]==new_target[k] for k in new_pred)
    p_first=sum(probe_pred[k] and probe_pred[k][0]==probe_target[k][0]
                for k in probe_pred)
    p_full=sum(probe_pred[k]==probe_target[k] for k in probe_pred)
    joined=dict(old_pred)
    if set(joined)&set(new_pred) or set(joined)&set(probe_pred) or set(new_pred)&set(probe_pred):
        raise AssertionError("S194 independent original/new32 provenance collision")
    joined.update(new_pred)
    return J7Retention584(
        actual_cpu_j7=True,
        old_checked=456,
        old_full_eos_correct=old_full,
        new_checked=128,
        new_full_eos_correct=new_full,
        old_first_lost=sum(old_pred[k][0]!=old_target[k][0] for k in old_pred),
        old_full_lost=456-old_full,
        new_first_lost=sum(new_pred[k][0]!=new_target[k][0] for k in new_pred),
        new_full_lost=128-new_full,
        protected_584_generation_digest=output_digest(joined),
        train32_checked=32,
        train32_first_correct=p_first,
        train32_full_eos_correct=p_full,
        train32_generation_digest=output_digest(probe_pred),
        serialized_fp16_reload_verified=physically_reloaded,
        frozen_parent_weights_unchanged=frozen_parent,
        no_reference_data_at_inference=captured_before_gold,
        source_only_train_no_QA_TEST=True,
        same_CPU_speed_vs_S187=speed_ratio,
        physical_all_inclusive_bpp=source_bpp,
        checkpoint_bytes_sha256=checkpoint_sha256,
    )

def actual_fp16_checkpoint_sha(paths: list[Path]) -> str:
    """Physical CPU load+serialize-reload EXACT FP16 tensor equality.

    We hash the persisted original source bytes, not a guessed model flag.
    This fingerprint is for BOTH frozen MPS S142 sidecar and S160 head.
    """
    from safetensors.torch import load_file,save_file
    import tempfile
    import torch
    if len(paths)!=2 or any(not p.is_file() for p in paths):
        raise AssertionError("S194 exactly S142 43M and S160 frozen FP16 inputs")
    h=hashlib.sha256()
    for p in paths:
        content=p.read_bytes()
        h.update(len(content).to_bytes(8,"big"))
        h.update(content)
        loaded=load_file(str(p),device="cpu")
        if not loaded or not all(torch.isfinite(v).all() for v in loaded.values() if v.is_floating_point()):
            raise AssertionError("S194 nonfinite/empty physical FP16 source")
        with tempfile.TemporaryDirectory() as folder:
            cp=Path(folder)/"physical-reload.safetensors"
            save_file(loaded,str(cp))
            reopened=load_file(str(cp),device="cpu")
            if set(loaded)!=set(reopened) or any(
                not torch.equal(loaded[k],reopened[k]) for k in loaded
            ):
                raise AssertionError("S194 serialized physical FP16 mismatch")
    return h.hexdigest()

def real_j7_baseline(argv: list[str], *, old_s87: Path):
    """Run full S190 real native CPU once. All capture before gold open."""
    from calibri2.s160 import real_parallel_slots_j7 as s160
    from calibri2.s182 import real_cpu_j7_shadow_guard as s182
    from calibri2.s190 import real_cpu_new_train_paraphrase_probe as s190
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    import torch

    if REAL_CAPTURE or s182.ALL_PRED or s190.PROBE_RECORDS is not None:
        raise AssertionError("S194 CPU real run must use a fresh isolated process")
    if torch.cuda.is_available() or sys.platform!="linux":
        raise AssertionError("S194 frozen CPU must run on GitHub Linux CPU")
    probes=s190.new_train_paraphrase_probes()
    probes_by_sha={sha_text(q):(q,a,g) for q,a,g in probes}
    original_decoder=s160.cached_full_cpu_parallel
    def capture(*args,**kwargs):
        q=args[6] if len(args)>6 else kwargs.get("question")
        result=original_decoder(*args,**kwargs)
        key=sha_text(q)
        if key in probes_by_sha:
            ids=list(result[0])
            if key in REAL_CAPTURE and REAL_CAPTURE[key]!=ids:
                raise AssertionError("S194 repeated real CPU J7 new32 generated different IDs")
            REAL_CAPTURE[key]=ids
        return result
    previous_argv=list(sys.argv)
    original_trace=old_s87.stat().st_size
    s160.cached_full_cpu_parallel=capture
    sys.argv=[str(argv[0]),*argv[1:]]
    try:
        s190.main()
    finally:
        sys.argv=previous_argv
        s160.cached_full_cpu_parallel=original_decoder
    if len(REAL_CAPTURE)!=32 or old_s87.stat().st_size!=original_trace:
        raise AssertionError("S194 32 actual neural CPU J7 and immutable original source required")
    if set(s182.ALL_PRED)!={"old","new","new_expected"}:
        raise AssertionError("S194 S187 old 456/new 128 FULL J7 captured evidence absent")
    old=s182.ALL_PRED["old"]
    new=s182.ALL_PRED["new"]
    expected128=s182.ALL_PRED["new_expected"]
    if len(old)!=456 or len(new)!=128 or len(expected128)!=128:
        raise AssertionError("S194 456+128 real CPU results incomplete")
    # No gold/answers accessed by capture above. Only posthoc now:
    ref=json.loads(old_s87.read_text())
    if ref.get("stage")!="S87_456_TRAIN_SERIALIZED_COMPRESSED_CPU_AUDIT":
        raise AssertionError("S194 original frozen S87 TRAIN target provenance invalid")
    targets={row["question_sha256"]:row["target_ids"] for row in ref["all_456_per_identity_results"]}
    if len(targets)!=456:
        raise AssertionError("S194 original target identities incomplete")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    probe_target={}
    for q,a,g in probes:
        ids=tok.encode(a,add_special_tokens=False)
        if len(ids)!=(1 if g=="numeric_new_wording" else 2):
            raise AssertionError("S194 tokenizer NEW TRAIN full EOS ABI mismatch")
        probe_target[sha_text(q)]=ids+[tok.eos_token_id]
    output=Path(argv[-1])
    s190_source=json.loads((output/"S190_REAL_CPU_32_NEW_TRAIN_PARAPHRASE_PROBES.json").read_text())
    if (s190_source.get("original_real_CPU_FULL_EOS_456")!=456
        or s190_source.get("new128_real_CPU_FULL_EOS")!=128
        or s190_source.get("NEW_TRAIN_full_EOS_correct")!=0
        or s190_source.get("NEW_TRAIN_first_correct")!=16):
        raise AssertionError("S194 S190 frozen CPU source vs protected baseline changed")
    return old,new,expected128,targets,REAL_CAPTURE,probe_target,tok.eos_token_id,s190_source

def one_step_with_actual_physical_cpu_gate(
    *, params, optimizer, train_one_step, physically_export_fp16,
    independent_real_cpu_j7_on_export,
):
    """S194 attaches actual CPU validator to S193 atomic training.

    Exporter must serialize candidate / recovered FP16 to two fresh
    paths. Evaluator MUST load THOSE physical files and run full 616 J7.
    Never substitute the pinned baseline report for candidate inference.
    """
    from calibri2.s193.atomic_one_step_584 import one_protected_step
    def verify_fresh_physical():
        paths=physically_export_fp16()
        if not isinstance(paths,(list,tuple)) or len(paths)!=2:
            raise AssertionError("S194 TWO physical FP16 exports required")
        paths=[Path(p) for p in paths]
        checkpoint_hash=actual_fp16_checkpoint_sha(paths)
        cert=independent_real_cpu_j7_on_export(paths)
        if not isinstance(cert,J7Retention584) or not cert.compatible():
            raise AssertionError("S194 genuine 584+32 CPU J7 certificate required")
        if cert.checkpoint_bytes_sha256!=checkpoint_hash:
            raise AssertionError("S194 CPU J7 proof came from different physical weights")
        return cert
    return one_protected_step(params,optimizer,train_one_step,verify_fresh_physical)


def main():
    from dataclasses import asdict
    from calibri2.s182.real_cpu_j7_shadow_guard import DENOM
    if len(sys.argv)<15:
        raise SystemExit("S194 requires exact full S190 pinned CPU J7 CLI")
    args=list(sys.argv)
    out=Path(args[-1])
    if (out/"S194_ACTUAL_REAL_CPU_J7_584_PLUS32_GATE.json").exists():
        raise FileExistsError("S194 must never overwrite a true CPU J7 proof")
    # Required source artifact is the real S193 preflight; no training claim.
    pinned=Path("inputs/s193/S193_ATOMIC_ONE_STEP_584_PLUS32_PREFLIGHT_NO_WEIGHTS.json")
    if not pinned.is_file():
        raise AssertionError("S194 pinned S193 safety source artifact required")
    source=json.loads(pinned.read_text())
    if (source.get("stage")!="S193_ONE_STEP_GUARD_PREFLIGHT_ONLY"
        or source.get("protected_full_and_first_token_train_identities")!=584
        or source.get("actual_43M_GPU_weights_trained_in_S193") is not False
        or source.get("parent_S192_run")!=43342581672
        or source.get("model_accepted") is not False):
        raise AssertionError("S194 S193 immutable 584+32 provenance mismatch")
    total_bytes=sum(Path(p).stat().st_size for p in (
        "research/calibri2/s194/actual_cpu_j7_bridge.py",
        "research/calibri2/s194/test_actual_cpu_j7_bridge.py"))
    bpp=source["physical_bpp_budget_charging_S193_sources"]+8*total_bytes/DENOM
    if bpp>.7:
        raise AssertionError("S194 physical all-in source code pushes beyond 0.7 bpp")
    # S190 exact argv tail: S142 FP16, S142 report, S160 slot FP16,
    # S160 report, S179 no-label trace, S87 original TRAIN audit, output.
    for ix,sub in ((-7,"S142_ACTUAL_43M_ONE_J7_FP16"),
                   (-5,"S160_REAL_PARALLEL_SLOT_FP16")):
        if sub not in str(args[ix]):
            raise AssertionError("S194 exact original physical S142 and S160 43M sources required")
    ckpt=actual_fp16_checkpoint_sha([Path(args[-7]),Path(args[-5])])
    (old,new,new_expected,old_gold,probes,probes_gold,eos,
     s190_source)=real_j7_baseline(args,old_s87=Path(args[-2]))
    if not s190_source["no_gold_or_SHA_at_inference_or_routing"]:
        raise AssertionError("S194 S190 had target/identity router leakage")
    cert=make_real_584_certificate(
        old_pred=old,old_target=old_gold,
        new_pred=new,new_target=new_expected,
        probe_pred=probes,probe_target=probes_gold,eos_id=eos,
        source_bpp=bpp,speed_ratio=1.0,  # exact frozen S187/S190 source baseline, never trained proposal
        checkpoint_sha256=ckpt,physically_reloaded=True,
        frozen_parent=True,captured_before_gold=True)
    if (not cert.compatible()
        or cert.train32_full_eos_correct!=s190_source["NEW_TRAIN_full_EOS_correct"]
        or cert.train32_first_correct!=s190_source["NEW_TRAIN_first_correct"]
        or cert.old_full_eos_correct!=456 or cert.new_full_eos_correct!=128):
        raise AssertionError("S194 real 616 CPU J7 raw generated ID verification mismatch")
    report={
        "stage":"S194_TRUE_FROZEN_CPU_J7_584_PLUS32_BASELINE_ADAPTER",
        "parent_S190_run":43340296813,
        "parent_S193_run":43343676818,
        "all_616_exact_real_CPU_generated_tokens_captured":True,
        "no_labels_in_neural_decoder_or_router":True,
        "posthoc_source_LABEL_scoring_only":True,
        "baseline_identical_original_S187_S190_not_candidate":True,
        "actual_43M_training_steps":0,
        "candidate_weight_update_executed":False,
        "S193_one_step_physical_J7_callback_connector_ready":True,
        "actual_one_step_candidate_J7_verification_pending":True,
        "source_only_no_QA_TEST_LIVE":True,
        "model_accepted":False,
        "checkpoint_fingerprint_sha256":ckpt,
        "physical_FP16_reload_verified":True,
        "physical_bpp_including_S194_source":bpp,
        "J7Retention584_frozen_baseline":asdict(cert),
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"S194_ACTUAL_REAL_CPU_J7_584_PLUS32_GATE.json").write_text(
        json.dumps(report,sort_keys=True,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items()
                      if k!="J7Retention584_frozen_baseline"},sort_keys=True),flush=True)
    print(json.dumps({"event":"S194_REAL_CPU_J7_584_PLUS32_FULL_EOS",
                      "456":cert.old_full_eos_correct,
                      "128":cert.new_full_eos_correct,
                      "new32_first":cert.train32_first_correct,
                      "new32_full_EOS":cert.train32_full_eos_correct,
                      "physical_bpp":bpp,
                      "candidate_training_executed":False}),flush=True)

if __name__=="__main__":
    main()
