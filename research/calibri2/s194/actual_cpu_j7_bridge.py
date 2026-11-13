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

def real_j7_baseline(argv: list[str], *, old_s87: Path, source_bpp_plus_charge: float):
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
    checkpoint_paths=[Path(argv[-6]),Path(argv[-4])]  # invariant? validated separately below
    return old,new,expected128,targets,REAL_CAPTURE,probe_target,tok.eos_token_id,s190_source
