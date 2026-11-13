"""S200: real CPU multistep NRA training, immutable staged FP16 candidates.

S199 proved S198 single-step gives 0/32 full+EOS despite 584/584 retention.
Train only the same four original precommitted NEW TRAIN examples.
Do not select or accept any snapshot without separately reloading it through
the full original 616 CPU J7, plus independent source-inclusive bpp and speed.
No QA, TEST, LIVE, Render, Supabase, APK, or model deployment.
"""
from __future__ import annotations

import hashlib
import json
import lzma
from pathlib import Path
import sys

import torch
from safetensors.torch import load, load_file, save
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s112.prompt_conditioned_first_token_mps import load_cpu_original
from calibri2.s132.progressive_core import ProgressiveRouteCore, IDS_KEY, WEIGHTS_KEY
from calibri2.s195_neuro_area.representation_area import NeuroRepresentationArea, PREFIX
from calibri2.s198.one_step_real_nra import (
    CHOSEN_INDICES, new_train_loss, validate_parent,
)
from calibri2.s190.real_cpu_new_train_paraphrase_probe import new_train_paraphrase_probes
from v176.operators import clear

STAGE = "S200_REAL_MULTISTEP_NRA_TRAIN_ONLY_UNACCEPTED"
S197_RUN = 43369398297
S199_VERIFIED_RUN = 43372509394
MILESTONES = (2, 4, 8)
LEARNING_RATE = 8e-4
SEED = 200013


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_candidate(*,original,area,out,step):
    """Only exact FP16 alternates; never overwrite parent or earlier step."""
    new_state = area.fp16_state()
    if set(k for k in original if k.startswith(PREFIX)) != set(new_state):
        raise AssertionError("S200 complete NRA layer tensors must be physically present")
    proposed = dict(original)
    proposed.update(new_state)
    if not all(torch.equal(proposed[k],original[k]) for k in original if not k.startswith(PREFIX)):
        raise AssertionError("S200 old core / Green Tunnel / grown adapter mutated")
    if not torch.any(proposed[PREFIX+"decoder.weight"] != original[PREFIX+"decoder.weight"]):
        raise AssertionError("S200 real neural gradient failed to change FP16")
    packet = lzma.compress(save(proposed), preset=6)
    fn = out/f"S200_UNACCEPTED_REAL_NRA_STEP_{step}_FP16.safetensors.lzma"
    if fn.exists():
        raise FileExistsError("S200 immutable step snapshot already exists")
    fn.write_bytes(packet)
    restored = load(lzma.decompress(fn.read_bytes()))
    if set(restored)!=set(proposed) or any(not torch.equal(restored[k],t) for k,t in proposed.items()):
        raise AssertionError("S200 physical FP16 package roundtrip not exact")
    if any(not torch.equal(restored[k],t) for k,t in original.items() if not k.startswith(PREFIX)):
        raise AssertionError("S200 old preserved full weights changed after physical reload")
    return {
        "step":step,
        "filename":str(fn),
        "lzma_bytes":len(packet),
        "sha256":sha(packet),
        "NRA_FP16_elements_changed":{
            k:int(torch.count_nonzero(restored[k]!=original[k]))
            for k in new_state
        },
        "full_616_CPU_J7_still_required":True,
        "accepted":False,
    }


def train(*,parent_package:Path,parent_report:Path,original_s93:Path,
          original_s93_side:Path,out:Path):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("S200 distinct immutable candidate directory required")
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    parent,original=validate_parent(compressed=parent_package,s197_report=parent_report)
    if original[WEIGHTS_KEY].shape!=(12,320,72) or len(original[IDS_KEY])!=72:
        raise AssertionError("S200 original rank1344 physically grown 46M ABI invalid")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    raw=s87.open_envelope(original_s93.read_bytes())
    side=load_file(str(original_s93_side),device="cpu")
    cpu=load_cpu_original(raw,side)
    model=cpu[0]
    core=ProgressiveRouteCore(rank=1344).eval().install(model)
    core.load_active_fp16({k:original[k] for k in (IDS_KEY,WEIGHTS_KEY)})
    area=NeuroRepresentationArea().eval()
    area.load_state_dict({
        k.removeprefix(PREFIX):v.float() for k,v in original.items()
        if k.startswith(PREFIX)},strict=True)
    area.install(model)
    area._active=True
    original_digests={k:sha(t.numpy().tobytes()) for k,t in original.items() if not k.startswith(PREFIX)}
    records=new_train_paraphrase_probes()
    chosen=[records[i] for i in CHOSEN_INDICES]
    if len(records)!=32 or len(chosen)!=4 or len({q for q,_,_ in chosen})!=4:
        raise AssertionError("S200 must train only four precommitted S198 NEW TRAIN tasks")
    opt=torch.optim.AdamW(area.parameters(),lr=LEARNING_RATE,weight_decay=0.0)
    snapshots=[]
    losses=[]
    try:
        model.eval().requires_grad_(False)
        for step in range(1,MILESTONES[-1]+1):
            opt.zero_grad(set_to_none=True)
            loss=new_train_loss(
                model=model,cpu=cpu,area=area,core=core,tok=tok,records=chosen)
            if not torch.isfinite(loss):
                raise AssertionError("S200 actual CPU loss nonfinite")
            loss.backward()
            grad=area.decoder.weight.grad
            if grad is None or not torch.isfinite(grad).all() or not torch.any(grad):
                raise AssertionError("S200 real decoder CPU gradients missing")
            norm=float(torch.nn.utils.clip_grad_norm_(area.parameters(),1.0))
            if not (norm>=0 and norm<float("inf")):
                raise AssertionError("S200 bad CPU gradient norm")
            opt.step()
            losses.append({"step":step,"pre_step_loss":float(loss.detach()),"grad_norm":norm})
            if step in MILESTONES:
                snapshots.append(save_candidate(original=original,area=area,out=out,step=step))
                print(json.dumps({
                    "event":"S200_PHYSICAL_UNACCEPTED_NRA_SNAPSHOT",
                    "step":step,"train_loss":float(loss.detach()),
                    "bytes":snapshots[-1]["lzma_bytes"],
                    "still_requires_full_J7":True
                },sort_keys=True),flush=True)
        if area.last_coverage!=tuple(range(12)) or area.forward_calls<8:
            raise AssertionError("S200 not connected to all 12 original CPU layers")
        for k,v in original_digests.items():
            if sha(original[k].numpy().tobytes())!=v:
                raise AssertionError("S200 frozen original core/Green Tunnel changed")
        if len(snapshots)!=len(MILESTONES):
            raise AssertionError("S200 2/4/8 checkpoint series missing")
        report={
            "stage":STAGE,
            "S197_immutable_616_parent_run":S197_RUN,
            "S199_single_step_rejected_no_gain_run":S199_VERIFIED_RUN,
            "parent_sha256":sha(parent_package.read_bytes()),
            "original_S197_protected_full_EOS":parent["real_original_full_EOS"]+parent["real_new128_full_EOS"],
            "actual_CPU_training_steps":MILESTONES[-1],
            "original_NEW_TRAIN_records_total":len(records),
            "precommitted_trained_count":len(chosen),
            "lr":LEARNING_RATE,
            "train_losses":losses,
            "immutable_candidate_snapshots":snapshots,
            "old_non_NRA_tensors_byte_exact":True,
            "real_12_layer_neural_gradient":True,
            "no_heldout_QA_TEST":True,
            "full_CPU_616_J7_protected_retention_NOT_YET_CHECKED":True,
            "independent_physical_bpp_AND_paired_CPU_speed_REQUIRED":True,
            "no_source_answer_or_question_SHA_at_inference":True,
            "Jessica_LIVE_Render_Supabase_APK_untouched":True,
            "model_accepted":False,
        }
        (out/"S200_REAL_NRA_MULTISTEP_TRAIN_REPORT.json").write_text(
            json.dumps(report,sort_keys=True,indent=2))
        print(json.dumps({
            "event":STAGE,"steps":MILESTONES[-1],
            "snapshots":len(snapshots),"full_616_CPU_J7_required":True,
            "model_accepted":False},sort_keys=True),flush=True)
    finally:
        area._active=False
        area.uninstall()
        core.uninstall()
        clear(cpu[1])
    return report


def main():
    if len(sys.argv)!=6:
        raise SystemExit("S200: S197.zlib S197_report.json S93.s61 S93_sidecar.safetensors outdir")
    train(parent_package=Path(sys.argv[1]),parent_report=Path(sys.argv[2]),
          original_s93=Path(sys.argv[3]),original_s93_side=Path(sys.argv[4]),
          out=Path(sys.argv[5]))


if __name__=="__main__":
    main()
