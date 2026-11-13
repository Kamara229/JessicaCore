"""S201: genuine separate full 616 native CPU J7 for each S200 NRA snapshot.

Reuse S199's unmodified S190→S187→S168 decoder and protected original 456+128
comparators. Patch ONLY which physically verified learned weights are installed,
and account for all S198→S201 runtime sources in conservative bpp. Each of
steps 2/4/8 is checked independently, not cherry-picked by expected answers.
No QA/TEST. Research candidate always unaccepted, no LIVE or deployment.
"""
from __future__ import annotations

import hashlib
import json
import lzma
from pathlib import Path
import sys
import torch
from safetensors.torch import load

from calibri2.s132.progressive_core import WEIGHTS_KEY
from calibri2.s195_neuro_area.representation_area import PREFIX
from calibri2.s196.core_growth_budget import PREVIOUS_PARAMS,NEXT_TOTAL_PARAMS
from calibri2.s199 import real_trained_nra_cpu_j7 as s199

STAGE="S201_REAL_S200_MULTISTEP_FULL_616_NATIVE_CPU_J7"
S200_RUN=43375165217
STEPS=(2,4,8)
SOURCE_FILES=[
    "research/calibri2/s198/one_step_real_nra.py",
    "research/calibri2/s198/test_one_step_real_nra.py",
    "research/calibri2/s199/real_trained_nra_cpu_j7.py",
    "research/calibri2/s199/test_real_trained_nra_cpu_j7.py",
    "research/calibri2/s200/multistep_real_nra_train.py",
    "research/calibri2/s200/test_multistep_real_nra_train.py",
    "research/calibri2/s201/real_s200_snapshots_full_j7.py",
    "research/calibri2/s201/test_real_s200_snapshots_full_j7.py",
]


def sha(blob:bytes)->str:
    return hashlib.sha256(blob).hexdigest()


def load_precommitted_s200(*,step,parent_zlib,parent_report,output_path):
    """Read exact physical precommitted candidate, no candidate-dependent routing."""
    import zlib
    if step not in STEPS:
        raise AssertionError("S201 only precommitted 2,4,8 original NEW TRAIN stages")
    record=json.loads(Path("inputs/s200/S200_REAL_NRA_MULTISTEP_TRAIN_REPORT.json").read_text())
    parent=json.loads(parent_report.read_text())
    base=parent_zlib.read_bytes()
    if (record.get("stage")!="S200_REAL_MULTISTEP_NRA_TRAIN_ONLY_UNACCEPTED"
        or record.get("S199_single_step_rejected_no_gain_run")!=43372509394
        or record.get("S197_immutable_616_parent_run")!=43369398297
        or record.get("model_accepted") is not False
        or record.get("original_S197_protected_full_EOS")!=584
        or record.get("old_non_NRA_tensors_byte_exact") is not True
        or record.get("parent_sha256")!=sha(base)
        or parent.get("stage")!="S197_ACTUAL_PHYSICAL_46M_NRA_REAL_CPU_616_J7"
        or parent.get("exact_protected_584_generated_ids_equal_S194") is not True
        or parent.get("checkpoint_runtime_physical",{}).get("runtime_packed_sha256")!=sha(base)):
        raise AssertionError("S201 exact validated immutable S197/S200 TRAIN provenance invalid")
    snapshots=record["immutable_candidate_snapshots"]
    if [x["step"] for x in snapshots]!=list(STEPS):
        raise AssertionError("S201 S200 precommitted checkpoints 2,4,8 missing")
    chosen=next(x for x in snapshots if x["step"]==step)
    filename=Path(chosen["filename"]).name
    if filename!=f"S200_UNACCEPTED_REAL_NRA_STEP_{step}_FP16.safetensors.lzma":
        raise AssertionError("S201 wrong S200 candidate physical filename")
    source=Path("inputs/s200")/filename
    blob=source.read_bytes()
    if (sha(blob)!=chosen["sha256"]
        or len(blob)!=chosen["lzma_bytes"]
        or chosen.get("accepted") is not False
        or chosen.get("full_616_CPU_J7_still_required") is not True):
        raise AssertionError("S201 S200 immutable step physical SHA/budget invalid")
    original=load(zlib.decompress(base))
    candidate=load(lzma.decompress(blob))
    if (set(original)!=set(candidate)
        or candidate[WEIGHTS_KEY].shape!=(12,320,72)
        or any(not torch.equal(original[k],candidate[k]) for k in original
               if not k.startswith(PREFIX))
        or not any(not torch.equal(original[k],candidate[k]) for k in original
                   if k.startswith(PREFIX))):
        raise AssertionError("S201 candidate altered frozen core/GT or no NRA learning")
    if output_path.exists():
        raise FileExistsError("S201 may not overwrite a physically loaded runtime")
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_bytes(blob)
    if sha(output_path.read_bytes())!=sha(blob):
        raise AssertionError("S201 persisted model changed")
    physical={
        "parent_package_bytes":len(base),
        "actual_single_runtime_lzma_bytes":len(blob),
        "actual_runtime_lzma_SHA256":sha(blob),
        "raw_all_FP16_bytes":len(lzma.decompress(blob)),
        "actual_runtime_FP16_sha256":sha(lzma.decompress(blob)),
        "new_NRA_FP16_train_changed": chosen["NRA_FP16_elements_changed"],
        "S200_pinned_CI_run":S200_RUN,
        "S200_step":step,
    }
    training={
        "actual_CPU_train_steps":step,
        "train_NRA_Fp16_changed_elements":chosen["NRA_FP16_elements_changed"],
    }
    return parent,training,candidate,physical


def physical_invoice(*,parent_report,physical,manifest_bytes,source_paths):
    """No free structural zero parameters or missing S198-S201 runtime sources."""
    del source_paths
    previous_bits=float(parent_report["all_in_physical_bpp"]["all_in_paid_bits"])
    old=int(physical["parent_package_bytes"])
    current=int(physical["actual_single_runtime_lzma_bytes"])
    sources=sum(Path(p).stat().st_size for p in SOURCE_FILES)
    if previous_bits<=0 or old<=0 or current<=0 or manifest_bytes<0:
        raise AssertionError("S201 invalid full physical byte invoice")
    all_bits=previous_bits+8*(current-old+sources+manifest_bytes)
    return {
        "all_in_bits":all_bits,
        "architectural_bpp":all_bits/NEXT_TOTAL_PARAMS,
        "conservative_bpp_excluding_zero_structural_growth":all_bits/(PREVIOUS_PARAMS+21_900),
        "replaced_S197_package_bytes":old,
        "persisted_lzma_runtime_bytes":current,
        "S198_S199_S200_S201_sources_charged":sources,
        "this_manifest_bytes_charged":manifest_bytes,
    }


def main():
    if len(sys.argv)<17:
        raise SystemExit("S201: STEP then exact original S199 616 native CPU CLI")
    step=int(sys.argv[1])
    if step not in STEPS:
        raise AssertionError("S201 unexpected step selection")
    original=sys.argv[:]
    source_original=s199.physical_and_parent
    budget_original=s199.physical_invoice
    stage_original=s199.S199_STAGE
    # S160 speed audits compare the original frozen S142 parent decoder against
    # a candidate decoder. An NRA learned at S200 step 8 may alter the parent's
    # output if left active in the baseline leg. Freeze NRA ONLY on the baseline
    # leg; keep it active on all candidate decoding and the S190 32 TRAIN probes.
    from calibri2.s190 import real_cpu_new_train_paraphrase_probe as s190
    from calibri2.s160 import real_parallel_slots_j7 as s160
    old_area_install=s199.NeuroRepresentationArea.install
    old_paired_speed=s190.BASE_SPEED
    installed_areas=[]
    def track_area_install(area,model):
        result=old_area_install(area,model)
        installed_areas.append(area)
        return result
    def parent_only_paired_speed(*args,**kwargs):
        native_parent_decode=s160.cached_full_cpu_with_neural_copy
        def frozen_parent_speed_decode(*a,**kw):
            if len(installed_areas)!=1 or not installed_areas[0]._active:
                raise AssertionError("S201 physical parent speed requires installed active candidate NRA")
            area=installed_areas[0]
            area._active=False
            try:
                return native_parent_decode(*a,**kw)
            finally:
                area._active=True
        s160.cached_full_cpu_with_neural_copy=frozen_parent_speed_decode
        try:
            return old_paired_speed(*args,**kwargs)
        finally:
            s160.cached_full_cpu_with_neural_copy=native_parent_decode
    s199.NeuroRepresentationArea.install=track_area_install
    s190.BASE_SPEED=parent_only_paired_speed
    s199.physical_and_parent=lambda parent_zlib,parent_report,candidate_zlib,train_report,destination: load_precommitted_s200(
        step=step,parent_zlib=parent_zlib,parent_report=parent_report,output_path=destination)
    s199.physical_invoice=physical_invoice
    s199.S199_STAGE=STAGE
    try:
        sys.argv=[sys.argv[0],*sys.argv[2:]]
        s199.main()
    finally:
        s199.physical_and_parent=source_original
        s199.physical_invoice=budget_original
        s199.S199_STAGE=stage_original
        s199.NeuroRepresentationArea.install=old_area_install
        s190.BASE_SPEED=old_paired_speed
        sys.argv=original
    out=Path(original[-1])
    old=out/"S199_TRAINED_NRA_REAL_616_CPU_J7_BPP.json"
    if not old.exists():
        raise AssertionError("S201 real full native CPU 616 manifest missing")
    report=json.loads(old.read_text())
    report["stage"]=STAGE
    report["S200_step"]=step
    report["physical_manifest_S201_only"]=True
    report["CPU_ratio_independently_paired_for_new_candidate"]=False
    report["model_accepted"]=False
    report["research_candidate_S199_j7_gate_pass"]=False
    # Charge actual final S201 manifest bytes (rather than intermediate JSON).
    parent_report=json.loads(Path("inputs/s197/S197_REAL_46M_NRA_616_CPU_J7.json").read_text())
    report_size=0
    for _ in range(40):
        report["total_physical_budget"]=physical_invoice(
            parent_report=parent_report,physical=report["physical_checkpoint"],
            manifest_bytes=report_size,source_paths=())
        report["J7_acceptance_guard"]["conservative_bpp_no_free_zeros_under_07"]=(
            report["total_physical_budget"]["conservative_bpp_excluding_zero_structural_growth"]<=0.7)
        report["J7_acceptance_guard"]["full_physical_bpp_architecture_under_07"]=(
            report["total_physical_budget"]["architectural_bpp"]<=0.7)
        payload=json.dumps(report,sort_keys=True,indent=2)
        if len(payload.encode())==report_size:
            break
        report_size=len(payload.encode())
    else:
        raise AssertionError("S201 final source-inclusive physical report size did not converge")
    destination=out/f"S201_STEP_{step}_FULL_616_REAL_CPU_J7.json"
    destination.write_text(payload)
    old.unlink()
    gate=report["J7_acceptance_guard"]
    print(json.dumps({
        "event":STAGE,"step":step,
        "old_456":report["full_456_correct_EOS"],
        "new128":report["full_128_correct_EOS"],
        "new32_first":report["32_NEW_TRAIN_first_correct"],
        "new32_full":report["32_NEW_TRAIN_full_EOS_correct"],
        "all_in_conservative_bpp":report["total_physical_budget"]["conservative_bpp_excluding_zero_structural_growth"],
        "all_584_full_EOS_preserved":gate["all_584_previous_answers_retain_full_EOS"],
        "independent_paired_CPU_speed_pending":True,
        "model_accepted":False},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
