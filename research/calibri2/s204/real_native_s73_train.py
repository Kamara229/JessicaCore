"""S204 real, isolated, TRAIN-only CPU NRA projection from S197 parent.

Authentic original 456 S87 TRAIN and their frozen S95 routes bind four
S73 protected first-token gradients to the *same* actual CPU 46M+GT+NRA
model that learns four precommitted S190 NEW TRAIN cases.

Candidate is ALWAYS unaccepted. Independent native CPU full+EOS 456+128+32
J7, source-inclusive bpp and paired CPU speed are REQUIRED AFTER this step.
No QA/TEST, no LIVE Jessica, Render, Supabase, main or APK modifications.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
from pathlib import Path

import torch
from safetensors.torch import load, save
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s99.guard_neural_copy_456_cpu import load_frozen_numeric_weights
from calibri2.s112.prompt_conditioned_first_token_mps import cohort, load_cpu_original
from calibri2.s94.physical_profile_bank import freeze_report
from calibri2.s111.prompt_conditioned_dual_tunnel import PromptConditionedDualTunnel
from calibri2.s126.full_layer_state_tunnel import FullLayerStateTunnel
from calibri2.s126.compact_rank4_q2 import load_compact_q2
from calibri2.s127.q2_prefill_gate_control import prefill_only_tunnel
from calibri2.s132.progressive_core import ProgressiveRouteCore, IDS_KEY, WEIGHTS_KEY
from calibri2.s160 import real_parallel_slots_j7 as s160
from calibri2.s158 import targeted_retention_gpu as s158
from calibri2.s195_neuro_area.representation_area import NeuroRepresentationArea, PREFIX
from calibri2.s198.one_step_real_nra import CHOSEN_INDICES, new_train_loss, validate_parent
from calibri2.s190.real_cpu_new_train_paraphrase_probe import new_train_paraphrase_probes
from calibri2.s204.protected_s73_train import precommitted_s73_original_train
from calibri2.s204.original_train_source_scope import make_original_s73_training_scope
from calibri2.s204.real_s73_guarded_step import real_four_s73_guarded_step
from v176.operators import clear

STAGE = "S204_REAL_PROTECTED_S73_CPU_NRA_STEP_UNACCEPTED"
PARENT_RUN = 43369398297
STRICT_J7_REQUIRED = True
NEW_TRAIN_LR = 8e-4
SEED = 204013
S158_INPUT_NAMES = (
    "S98_copy_net","S98_report","S95_router","S95_report",
    "S94_bank","S94_report","S93_CB61","S93_sidecar",
    "S99_full456_report","S101_full456_report",
    "S122_curriculum_report","S112_report",
    "S126Q_physical","S126Q_report",
    "S87_report","S91_report","S93_report",
    "S132_parent_sidecar","S132_parent_report","S131_parent_sidecar",
    "S142_parent_sidecar","S142_parent_guard",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="S204 genuine TRAIN-only protected NRA gradient on CPU")
    for name in S158_INPUT_NAMES:
        p.add_argument(name, type=Path)
    p.add_argument("S197_parent_package", type=Path)
    p.add_argument("S197_parent_report", type=Path)
    p.add_argument("out", type=Path)
    return p.parse_args(argv)


def train_real_204(a):
    if a.out.exists() and any(a.out.iterdir()):
        raise FileExistsError("S204 only unique immutable research artifact directory")
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    # Verify both physical S142 and the approved immutable S197 584 baseline.
    _guard, side, q2, _inherited = s158.check_sources(a)
    parent, original = validate_parent(
        compressed=a.S197_parent_package,s197_report=a.S197_parent_report)
    if (original[WEIGHTS_KEY].shape!=(12,320,72)
        or len(original[IDS_KEY])!=72
        or parent["real_original_full_EOS"]!=456
        or parent["real_new128_full_EOS"]!=128):
        raise AssertionError("S204 must start from unchanged real S197 full/EOS parent")
    frozen_sha = {k:hashlib.sha256(v.numpy().tobytes()).hexdigest()
                  for k,v in original.items() if not k.startswith(PREFIX)}
    _p94,_p95,_p98,bank,router,_copy_net = load_frozen_numeric_weights(a)
    tok = AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    reports={k:freeze_report(getattr(a,k+"_report")) for k in ("S87","S91","S93")}
    old,old_ids,old_routes=cohort(tok,reports,router)
    pinned=precommitted_s73_original_train(old)
    if len(pinned)!=4 or len(old_ids)!=456:
        raise AssertionError("S204 original 456 S73 source provenance absent")
    cpu=load_cpu_original(s87.open_envelope(a.S93_CB61.read_bytes()),side)
    model=cpu[0]
    bridge=full=expansion=area=None
    try:
        bridge=PromptConditionedDualTunnel(cpu[6]).eval().install(model)
        bridge.requires_grad_(False)
        full=FullLayerStateTunnel(width=320,depth=12,rank=4).eval()
        load_compact_q2(full,q2)
        full.requires_grad_(False)
        full.install(model)
        expansion=(ProgressiveRouteCore(rank=1344).eval()
                   .bind_original_route_bridge(bridge).install(model))
        expansion.load_active_fp16({k:original[k] for k in (IDS_KEY,WEIGHTS_KEY)})
        expansion.requires_grad_(False)
        area=NeuroRepresentationArea().eval()
        area.load_state_dict(
            {k.removeprefix(PREFIX):v.float() for k,v in original.items()
             if k.startswith(PREFIX)},strict=True)
        area.install(model)
        area._active=True
        model.eval().requires_grad_(False)
        core_sha=hashlib.sha256(original[WEIGHTS_KEY].numpy().tobytes()).hexdigest()
        scope=make_original_s73_training_scope(
            old_456=old,source_routes=old_routes,cpu=cpu,tok=tok,bank=bank,
            bridge=bridge,full=full,expansion=expansion)
        records=new_train_paraphrase_probes()
        chosen=[records[i] for i in CHOSEN_INDICES]
        if len(records)!=32 or len(chosen)!=4:
            raise AssertionError("S204 must use predeclared original S190 novel32 TRAIN")
        optimizer=torch.optim.AdamW(area.parameters(),lr=NEW_TRAIN_LR,weight_decay=0)
        def novel_train_loss():
            pieces=[]
            for record in chosen:
                q=record[0]
                # S190 source-only TRAIN does not use gold to choose route.
                features=torch.as_tensor(
                    s160.prompt_only_vector(tok,q),dtype=torch.float32)
                s160.apply_profile(cpu[1],cpu[6],"S93",bank,cpu[7],cpu[8])
                # S198.new_train_loss already wraps expansion.prefill_only.
                # Do not nest the same KV prefill hook for the real model.
                with prefill_only_tunnel(model,full),bridge.prompt_scope("S93",features):
                    pieces.append(new_train_loss(
                        model=model,cpu=cpu,area=area,core=expansion,
                        tok=tok,records=[record]))
            return torch.stack(pieces).mean()
        report=real_four_s73_guarded_step(
            original_456=old,source_routes=old_routes,source_scope=scope,
            model=model,cpu=cpu,tok=tok,area=area,optimizer=optimizer,
            backward_new32_train_loss=novel_train_loss)
        if (area.last_coverage!=tuple(range(12))
            or area.forward_calls<8
            or hashlib.sha256(original[WEIGHTS_KEY].numpy().tobytes()).hexdigest()!=core_sha):
            raise AssertionError("S204 frozen 46M core or 12-layer NRA learning drift")
        physical=dict(original)
        physical.update(area.fp16_state())
        if (set(physical)!=set(original)
            or any(not torch.equal(physical[k],original[k])
                   for k in original if not k.startswith(PREFIX))
            or not any(not torch.equal(physical[k],original[k])
                       for k in original if k.startswith(PREFIX))):
            raise AssertionError("S204 no real NRA update or frozen legacy weights altered")
        for k,digest in frozen_sha.items():
            if hashlib.sha256(original[k].numpy().tobytes()).hexdigest()!=digest:
                raise AssertionError("S204 original Green Tunnel / core corrupted")
        data=lzma.compress(save(physical),preset=6)
        path=a.out/"S204_UNACCEPTED_PROTECTED_NRA_ONE_STEP_FP16.safetensors.lzma"
        if path.exists():
            raise FileExistsError("S204 immutable candidate must not overwrite")
        path.write_bytes(data)
        restored=load(lzma.decompress(path.read_bytes()))
        if set(restored)!=set(physical) or any(
            not torch.equal(restored[k],v) for k,v in physical.items()):
            raise AssertionError("S204 physical candidate reload FP16 mismatch")
        doc={
            "stage":STAGE,
            "S197_immutable_parent_CI_run":PARENT_RUN,
            "parent_package_sha256":hashlib.sha256(a.S197_parent_package.read_bytes()).hexdigest(),
            "physical_candidate_sha256":hashlib.sha256(data).hexdigest(),
            "candidate_filename":path.name,
            "candidate_lzma_bytes":len(data),
            "source_original_456_full_EOS":456,
            "source_new128_full_EOS":128,
            "source_original_S73_TRAIN_ids":[k for k,_ in pinned],
            "frozen_source_routes":[old_routes[k] for k,_ in pinned],
            "trained_novel32_TRAIN_indices":list(CHOSEN_INDICES),
            "actual_CPU_neural_gradients_and_Adam_step":True,
            "genuine_old_token_gradients":4,
            "frozen_non_NRA_tensors_equal":True,
            "physical_roundtrip_verified":True,
            "s204_train_audit":report,
            "actual_native_456_128_32_CPU_J7_still_required":True,
            "independent_paired_CPU_speed_and_source_inclusive_bpp_required":True,
            "no_QA_TEST_LIVE_or_serving_changes":True,
            "model_accepted":False,
        }
        (a.out/"S204_PROTECTED_S73_REAL_CPU_TRAIN_ONLY.json").write_text(
            json.dumps(doc,indent=2,sort_keys=True))
        print(json.dumps({
            "event":STAGE,
            "original_protected_first_token_grads":4,
            "fp16_update":True,
            "candidate_bytes":len(data),
            "native_full_CPU_J7_PENDING":True,
            "model_accepted":False},sort_keys=True),flush=True)
        return doc
    finally:
        if area is not None:
            area._active=False
            area.uninstall()
        if expansion is not None:
            expansion.uninstall()
        if full is not None:
            full.uninstall()
        if bridge is not None:
            bridge.uninstall()
        clear(cpu[1])


def main():
    a=parse_args()
    train_real_204(a)


if __name__=="__main__":
    main()
