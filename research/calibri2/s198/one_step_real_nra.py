"""S198: ONE authentic CPU gradient step into S197 neural representation weights.

The actual frozen S101 36M dense-equivalent CPU core, original Green Tunnel
and S197 rank1344 main adapter are physically instantiated. All original
parameters are frozen. The S195 NRA alone receives gradients from original
S190 NEW TRAIN questions. No synthetic teacher, mock model, pseudo-training,
QA/TEST, source-oracle answer lookup, LIVE or external deployment.

This is an independently audited *candidate* only. A subsequent physically
reloaded full 616 CPU J7 must establish gains, forgetting and actual speed
before any acceptance; model_accepted never becomes true here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import zlib

import torch
from torch.nn import functional as F
from safetensors.torch import load,load_file,save
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s112.prompt_conditioned_first_token_mps import load_cpu_original
from calibri2.s132.progressive_core import ProgressiveRouteCore,IDS_KEY,WEIGHTS_KEY
from calibri2.s160.real_parallel_slots_j7 import prompt_tokens
from calibri2.s190.real_cpu_new_train_paraphrase_probe import new_train_paraphrase_probes
from calibri2.s195_neuro_area.representation_area import NeuroRepresentationArea,PREFIX as NRA_PREFIX
from calibri2.s196.core_growth_budget import S135_PARAMS, NEXT_TOTAL_PARAMS
from v176.operators import clear

STAGE="S198_GENUINE_ONE_CPU_NRA_GRADIENT_STEP_TRAIN_ONLY"
S197_SOURCE_RUN=43369205584
CHOSEN_INDICES=(0,1,2,24)  # predetermined, diverse owner bindings and numeric; no adaptive selection
TRAIN_LR=0.0003


def physical_digest(blob:bytes)->str:
    return hashlib.sha256(blob).hexdigest()


def validate_parent(*,compressed:Path,s197_report:Path):
    parent=json.loads(s197_report.read_text())
    data=compressed.read_bytes()
    if (parent.get("stage")!="S197_ACTUAL_PHYSICAL_46M_NRA_REAL_CPU_616_J7"
        or parent.get("real_original_full_EOS")!=456
        or parent.get("real_new128_full_EOS")!=128
        or parent.get("real_new32_full_EOS")!=0
        or parent.get("exact_protected_584_generated_ids_equal_S194") is not True
        or parent.get("no_learned_weight_or_original_Green_Tunnel_modified") is not True
        or parent.get("main_core_architecture_params")!=S135_PARAMS
        or parent.get("total_architecture_params")!=NEXT_TOTAL_PARAMS
        or parent.get("checkpoint_runtime_physical",{}).get("runtime_packed_sha256")!=physical_digest(data)
        or parent.get("all_in_physical_bpp",{}).get("conservative_excluding_new_zero_growth_bpp",1)>.7
        or parent.get("model_accepted") is not False):
        raise AssertionError("S198 needs immutable actual S197 physical 584+32 J7 source")
    raw=zlib.decompress(data)
    weights=load(raw)
    if (weights[WEIGHTS_KEY].shape!=(12,320,72)
        or len(weights[IDS_KEY])!=72
        or len([x for x in weights if x.startswith(NRA_PREFIX)])!=6):
        raise AssertionError("S198 exact S197 expanded+NRA physical ABI missing")
    if torch.count_nonzero(weights[WEIGHTS_KEY][:,:,48:]):
        raise AssertionError("S198 new rank1344 channels were not untrained zeros")
    return parent,weights


def new_train_loss(*,model,cpu,area,core,tok,records):
    """Teacher forcing through TRUE frozen core+Green Tunnel+NRA, no source answers in inference."""
    model.eval()
    model.requires_grad_(False)
    area.train()
    all_losses=[]
    for q,expected,group in records:
        prompt=prompt_tokens(tok,q)
        gold=tok.encode(expected,add_special_tokens=False)+[tok.eos_token_id]
        if not prompt or len(gold)!=(2 if group=="numeric_new_wording" else 3):
            raise AssertionError("S198 original S190 NEW TRAIN token ABI drift")
        ids=torch.tensor([prompt+gold[:-1]],dtype=torch.long)
        clear(cpu[1])
        # True forward through actual compressed 36M model, rank1344 adapter,
        # existing internal Green Tunnel and all 12 newly appended area hooks.
        with core.prefill_only(model):
            result=model(input_ids=ids,past_key_values=None,use_cache=True,output_hidden_states=True)
        if not area.forward_calls or area.last_coverage!=tuple(range(12)):
            raise AssertionError("S198 NRA not connected to real 12-layer backprop path")
        hidden=result.hidden_states[-1][0]
        logits=result.logits[0]
        pieces=[]
        for step,gold_id in enumerate(gold):
            offset=len(prompt)-1+step
            h=hidden[offset]
            final=logits[offset]+F.linear(cpu[2](h) if step==0 else cpu[3](h),cpu[5])
            target=torch.tensor([gold_id],dtype=torch.long)
            ce=F.cross_entropy(final.unsqueeze(0),target)
            pieces.append(ce*(1.0 if step==0 else 1.4))
        all_losses.append(torch.stack(pieces).mean())
        del result
    return torch.stack(all_losses).mean()


def train_once(*,parent_package:Path,s197_report:Path,original_s93:Path,
               original_s93_side:Path,out:Path):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("S198 distinct immutable experiment output required")
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(198013)
    parent,original=validate_parent(compressed=parent_package,s197_report=s197_report)
    from calibri2.s132.growth_schedule import architecture_params
    if architecture_params(1344)!=S135_PARAMS:
        raise AssertionError("S198 rank1344 architecture drift")
    original_sha={k:physical_digest(v.numpy().tobytes()) for k,v in original.items()}
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
        k.removeprefix(NRA_PREFIX):v.float() for k,v in original.items()
        if k.startswith(NRA_PREFIX)},strict=True)
    area.install(model)
    area._active=True
    before={k:v.detach().clone() for k,v in area.fp16_state().items()}
    records=new_train_paraphrase_probes()
    chosen=[records[i] for i in CHOSEN_INDICES]
    if len(records)!=32 or len({q for q,_a,_g in chosen})!=len(chosen):
        raise AssertionError("S198 must use original immutable S190 NEW TRAIN only")
    optimizer=torch.optim.AdamW(area.parameters(),lr=TRAIN_LR,weight_decay=0.0)
    try:
        model.eval().requires_grad_(False)
        optimizer.zero_grad(set_to_none=True)
        loss=new_train_loss(
            model=model,cpu=cpu,area=area,core=core,tok=tok,records=chosen)
        if not torch.isfinite(loss):
            raise AssertionError("S198 real CPU teacher-forcing loss nonfinite")
        loss.backward()
        decoder_grad=area.decoder.weight.grad
        if (decoder_grad is None or not bool(torch.isfinite(decoder_grad).all())
            or not bool(torch.count_nonzero(decoder_grad))):
            raise AssertionError("S198 real frozen 36M/46M core did not backpropagate to NRA decoder")
        grad_norm=float(torch.nn.utils.clip_grad_norm_(area.parameters(),1.0))
        optimizer.step()
        trained=area.fp16_state()
        updates={
            name:int(torch.count_nonzero(trained[name]!=before[name]).item())
            for name in before
        }
        if updates[NRA_PREFIX+"decoder.weight"]<1:
            raise AssertionError("S198 float gradients did not survive physical FP16 serialization")
        for name,val in original.items():
            if name.startswith(NRA_PREFIX):
                continue
            if physical_digest(val.numpy().tobytes())!=original_sha[name]:
                raise AssertionError("S198 immutable old learned tensor mutated")
        candidate=dict(original)
        candidate.update(trained)
        if any(not torch.equal(original[k],candidate[k]) for k in original
               if not k.startswith(NRA_PREFIX)):
            raise AssertionError("S198 old core, S142 and GT mutated on optimizer step")
        compressed=zlib.compress(save(candidate),9)
        filepath=out/"S198_UNACCEPTED_ONE_TRUE_CPU_NRA_STEP_46M_FP16.safetensors.zlib"
        filepath.write_bytes(compressed)
        physical=load(zlib.decompress(filepath.read_bytes()))
        if (set(candidate)!=set(physical)
            or any(not torch.equal(candidate[k],physical[k]) for k in candidate)
            or not any(not torch.equal(original[k],physical[k]) for k in original
                       if k.startswith(NRA_PREFIX))):
            raise AssertionError("S198 candidate physical FP16 roundtrip failed")
        print(json.dumps({"event":"S198_ONE_ACTUAL_CPU_NEURAL_GRADIENT",
            "train_loss":float(loss.detach()),"grad_norm":grad_norm,
            "new_train_samples":len(chosen),
            "NRA_changed_fp16_elements":updates,
            "frozen_core_tensors_unchanged":True},sort_keys=True),flush=True)
        report={
            "stage":STAGE,
            "source_S197_pinned_run":S197_SOURCE_RUN,
            "parent_package_SHA256":physical_digest(parent_package.read_bytes()),
            "candidate_package_SHA256":physical_digest(compressed),
            "candidate_package_bytes":len(compressed),
            "parent_package_bytes":parent_package.stat().st_size,
            "main_core_architecture_params":S135_PARAMS,
            "NRA_architecture_params":area.parameter_count(),
            "total_architecture_params":NEXT_TOTAL_PARAMS,
            "actual_CPU_train_steps":1,
            "actual_CPU_teacher_forcing_grad_nonzero":True,
            "actual_original_CPU_model_and_Green_Tunnel_executed":True,
            "actual_rank1344_adapter_installed":True,
            "real_NRA_layer_coverage":list(area.last_coverage),
            "original_32_new_TRAIN_q_count":len(records),
            "precommitted_trained_NEW_TRAIN_count":len(chosen),
            "train_NRA_Fp16_changed_elements":updates,
            "gradient_norm":grad_norm,
            "train_teacher_loss":float(loss.detach()),
            "all_legacy_non_NRA_tensors_byte_identical":True,
            "source_S197_original_456_full":parent["real_original_full_EOS"],
            "source_S197_new128_full":parent["real_new128_full_EOS"],
            "independent_postupdate_real_CPU_J7_616_REQUIRED":True,
            "no_claim_of_new_correct_answers":True,
            "model_accepted":False,
            "no_QA_TEST_LIVE":True,
        }
        (out/"S198_ONE_CPU_NRA_TRAIN_REPORT.json").write_text(
            json.dumps(report,indent=2,sort_keys=True))
    finally:
        area._active=False
        area.uninstall()
        core.uninstall()
        clear(cpu[1])
    return report


def main():
    if len(sys.argv)!=6:
        raise SystemExit("S198 train CLI: parent_S197.zlib parent_report.json S93_CB61 S93_sidecar output")
    train_once(parent_package=Path(sys.argv[1]),s197_report=Path(sys.argv[2]),
        original_s93=Path(sys.argv[3]),original_s93_side=Path(sys.argv[4]),
        out=Path(sys.argv[5]))


if __name__=="__main__":
    main()
