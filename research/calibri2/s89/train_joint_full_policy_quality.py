"""S89: train for NEW strict full/EOS-correct answers, not just S87 identity.

RESEARCH ONLY. Input examples are pre-existing S30/S54/history/S69/S73/S82
TRAIN only. An immutable, separately generated 456-TRAIN S87 audit selects
WRONG-but-first-correct new training rows; the independent QA remains sealed.

S87 optimized the bare backbone logits without its active first/later
controllers or generic context pointer. S89 differentiates through the actual
deployed *full-policy* scoring path while jointly updating the two existing,
physically paid CB59 dictionaries and a tiny internal Green Tunnel. Frozen
zero gates retain S88's inference speed. Every candidate is *repacked*
into a physical CB61 source and reloaded on CPU for a TRAIN preflight.
Nothing here accepts or deploys a model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_PARAMS
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s78 import mps_36m_feature_cached_late_rank16_restoration as s78
from calibri2.s82 import real_mps_generic_token_context_copy as s82
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s87 import pilot_joint_core_tunnel as s87pilot
from calibri2.s87.internal_green_tunnel import (
    IntegratedGreenTunnel, select_paid_core_codebooks)
from calibri2.s88.zero_gate_fastpath import install_verified_fastpath
from v176.operators import clear
from v185.codec import replace_discrete

SEED=890089
STEPS=36
CHECKPOINTS=(12,24,36)
CORE_LR=0.00025
TUNNEL_LR=0.002
MAX_BPP=0.7


def hash_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def references(tok, old_report):
    existing=old_report["all_456_per_identity_results"]
    if len(existing)!=456 or len({x["id"] for x in existing})!=456:
        raise AssertionError("S89 requires complete frozen S87 456 TRAIN audit")
    prior={x["id"]:x for x in existing}
    found={}
    for group,pairs in s87.families(tok).items():
        for pair in pairs:
            for side in ("left","right"):
                record=pair[side]
                key=f'{group}/{pair["pair_id"]}/{side}'
                gold=tok.encode(record["answer"],add_special_tokens=False)+[
                    tok.eos_token_id]
                if key not in prior or prior[key]["target_ids"]!=gold:
                    raise AssertionError("S89 frozen old target identity drift: "+key)
                if hash_bytes(record["question"].encode())!=prior[key]["question_sha256"]:
                    raise AssertionError("S89 frozen 456 question SHA drift: "+key)
                found[key]={**record,"id":key,"family":group,
                            "expected":gold,"prior":prior[key]}
    if set(found)!=set(prior):
        raise AssertionError("S89 no TRAIN cohort expansion or omission allowed")
    # S82_new is the ONLY cohort for intentional new gains. Select complete
    # paired contrastive prompts, not answer strings from any QA set.
    candidates={}
    for k,r in found.items():
        x=r["prior"]
        if (r["family"]=="S82_new" and x["old_first"] and
                not x["old_full"] and len(r["expected"])==3):
            pair_id=k.rsplit("/",1)[0]
            candidates.setdefault(pair_id,[]).append(r)
    # Prefer swaps where BOTH members are wrong on full response, so
    # memorizing a single answer across both contrasts cannot pass.
    ordered=sorted(candidates.items(),key=lambda kv:(-len(kv[1]),kv[0]))
    selected_pairs=ordered[:6]
    targets=[r for _,group in selected_pairs for r in group]
    if len(selected_pairs)!=6 or len(targets)<8:
        raise AssertionError("S89 too few eligible independently frozen TRAIN errors")
    protected=[]
    rng=random.Random(SEED)
    for family in ("S30","S54","history","S69","S73","S82_new"):
        eligible=[r for r in found.values()
                  if r["family"]==family and r["prior"]["old_full"]
                  and r["id"] not in {t["id"] for t in targets}]
        if not eligible:
            raise AssertionError("S89 no protected correct examples in "+family)
        protected.extend(rng.sample(eligible,min(5,len(eligible))))
    if len(protected)<24:
        raise AssertionError("S89 insufficient interleaved protected rehearsal")
    return found,targets,protected


def policy_loss(model, tok, row, first, later, pointer, head,
                *, protected:bool):
    """Teacher-forcing through the ACTUAL first/later+pointer output policy.

    The last logit for the prompt predicts gold[0]. Subsequent gold-prefix
    positions predict gold[1], and EOS. This exactly matches inference's
    conditional second-token pointer and frozen controllers, unlike S87.
    """
    prompt=tok.apply_chat_template(
        [{"role":"user","content":row["question"]}],
        tokenize=False,add_generation_prompt=True)
    context=tok.encode(prompt,add_special_tokens=False)
    gold=row["expected"]
    if not context or len(gold)!=3:
        raise AssertionError("Only original two-token + EOS TRAIN authorized")
    ids=torch.tensor([context+gold[:-1]],dtype=torch.long,device="mps")
    result=model(input_ids=ids,use_cache=False,output_hidden_states=True)
    hidden=result.hidden_states[-1][0]
    base=result.logits[0]
    eligible=s82.candidate_positions(tok,context)
    source_ids=torch.tensor([context[i] for i in eligible],
                            dtype=torch.long,device="mps")
    losses=[]
    margin_losses=[]
    for step,token in enumerate(gold):
        position=len(context)-1+step
        h=hidden[position]
        total=base[position]+F.linear(first(h) if step==0 else later(h),head)
        if step==1 and eligible and int(total.detach().argmax())!=tok.eos_token_id:
            # Causal full-sequence prefix states match the first decode pass.
            keys=hidden[torch.tensor(eligible,device="mps")]
            extra=pointer.full_vocab_bonus(h,keys,source_ids)
            total=total.index_add(0,source_ids,extra)
        label=torch.tensor([token],device="mps",dtype=torch.long)
        ce=F.cross_entropy(total[None,:],label)
        # Weight known weak second-token and EOS completion instead of
        # optimizing S87's raw core logits against the wrong output policy.
        weight=(0.5,2.5,1.5)[step]
        losses.append(ce*weight)
        if protected:
            best=total.masked_fill(
                torch.arange(total.shape[0],device="mps")==token,
                -1.e8).max()
            margin=total[token]-best
            margin_losses.append(F.softplus(0.9-margin))
    objective=sum(losses)/sum((0.5,2.5,1.5))
    if protected:
        objective=0.65*objective+1.4*torch.stack(margin_losses).mean()
    return objective


def frozen_controllers(weights):
    first,later=s78.controllers(weights)
    ptr=s87.s82.GenericContextPointer()
    ptr.load_state_dict({
        k.removeprefix("generic_pointer."):v for k,v in weights.items()
        if k.startswith("generic_pointer.")},strict=True)
    return (first.to("mps").eval().requires_grad_(False),
            later.to("mps").eval().requires_grad_(False),
            ptr.to("mps").eval().requires_grad_(False))


def materialize_candidate(original_raw, sources, tunnel, s87_side, out, step):
    raw_new,mutation=replace_discrete(CB59(original_raw),sources,strict=False)
    packed=s87pilot.lossless_cb61(raw_new)
    prefix=f"S89_UNACCEPTED_STEP_{step}"
    source_path=out/(prefix+"_SOURCE_CB61.s61")
    side_path=out/(prefix+"_JOINT.safetensors")
    source_path.write_bytes(packed)
    side=dict(s87_side)
    for name,t in tunnel.state_dict().items():
        side["internal_tunnel."+name]=t.detach().cpu().contiguous()
    save_file(side,str(side_path))
    reloaded=load_file(str(side_path))
    if set(reloaded)!=set(s87_side):
        raise AssertionError("S89 cannot create new parameters")
    for name in s87_side:
        if not name.startswith("internal_tunnel.") and not torch.equal(
                s87_side[name],reloaded[name]):
            raise AssertionError("S89 changed frozen S82 controller")
    if s87.open_envelope(source_path.read_bytes())!=raw_new:
        raise AssertionError("S89 physical source does not roundtrip")
    if mutation["actually_changed_FP16_codebook_cells"]<1:
        raise AssertionError("S89 core dictionaries did not really change")
    return raw_new,side,source_path,side_path,mutation


def sample_guard(raw,side,tok,targets,protected,prior):
    state=s87.build_cpu(raw,side,True)
    model,sources,first,later,pointer,head,internal=state
    install_verified_fastpath(internal)
    ids=list({r["id"]:r for r in targets+protected}.values())
    gains=[];losses=[];first_losses=[];updated=[]
    try:
        for row in ids:
            output=s82.full_cpu_generate(model,sources,first,later,head,
                                         tok,row["question"],pointer)
            expected=row["expected"]
            old=row["prior"]
            full=output==expected
            first_ok=bool(output) and output[0]==expected[0]
            if old["old_full"] and not full: losses.append(row["id"])
            if old["old_first"] and not first_ok: first_losses.append(row["id"])
            if full and not old["old_full"]: gains.append(row["id"])
            if output!=old["S87_ids"]: updated.append(row["id"])
    finally:
        internal.uninstall()
        del state,model
    return dict(gained=gains,lost_full=losses,lost_first=first_losses,
                changed=updated,evaluated=len(ids))


def stable_manifest(meta,packed,side):
    for _ in range(40):
        encoded=json.dumps(meta,indent=2,sort_keys=True).encode()
        total=len(packed)+len(side)+len(encoded)
        fields={"manifest_bytes":len(encoded),
                "all_inclusive_physical_bytes":total,
                "all_inclusive_physical_bpp":f"{8*total/SOURCE_PARAMS:.12f}"}
        if all(meta.get(k)==v for k,v in fields.items()):break
        meta.update(fields)
    encoded=json.dumps(meta,indent=2,sort_keys=True).encode()
    if meta["all_inclusive_physical_bytes"]!=len(packed)+len(side)+len(encoded):
        raise AssertionError("S89 physical manifest size did not converge")
    if len(packed)+len(side)+len(encoded)>MAX_BPP*SOURCE_PARAMS/8:
        raise AssertionError("S89 exceeds all-inclusive 0.7 bpp")
    return encoded


def main():
    p=argparse.ArgumentParser()
    p.add_argument("S87_physical_CB61",type=Path)
    p.add_argument("S87_safetensors",type=Path)
    p.add_argument("S87_manifest",type=Path)
    p.add_argument("S87_complete_456_guard",type=Path)
    p.add_argument("S82_original_safetensors",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if not torch.backends.mps.is_available():
        raise RuntimeError("S89 requires REAL MPS gradients: no CPU fallback")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    old_bytes=a.S87_physical_CB61.read_bytes()
    old_raw=s87.open_envelope(old_bytes)
    old_side_raw=a.S87_safetensors.read_bytes()
    old_side=load_file(str(a.S87_safetensors))
    old_meta=json.loads(a.S87_manifest.read_text())
    reference=json.loads(a.S87_complete_456_guard.read_text())
    baseline_s82=a.S82_original_safetensors.read_bytes()
    if (hash_bytes(old_bytes)!=old_meta["source_CB61_physical_sha256"] or
        hash_bytes(old_raw)!=old_meta["source_sha256"] or
        hash_bytes(old_side_raw)!=old_meta["prototype_sidecar_sha256"] or
        hash_bytes(baseline_s82)!=old_meta["old_S82_sidecar_sha256"] or
        hash_bytes(old_raw)!=reference["source_new_cb59_sha256"] or
        not reference["full_safety_guard"] or
        reference["all_inclusive_bpp"]>MAX_BPP):
        raise AssertionError("S89 S87/S82 source or old 456 audit provenance drift")

    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    rows,targets,protected=references(tok,reference)
    student,sources=create_small_paid_model(
        CB59(old_raw),build_config(s87.MODEL_SPEC))
    student.to("mps").eval().requires_grad_(False)
    inner=IntegratedGreenTunnel(layers=(3,9),rank=8).to("mps").eval()
    inner.load_state_dict({k.removeprefix("internal_tunnel."):v for k,v in old_side.items()
                           if k.startswith("internal_tunnel.")},strict=True)
    for block in inner.blocks.values():
        block.gate.requires_grad_(False)
    install_verified_fastpath(inner)
    inner.install(student)
    core=select_paid_core_codebooks(sources,(3,9))
    core_params=[s.centroids for s in core.values()]
    tunnel_params=[p for p in inner.parameters() if p.requires_grad]
    if len(core_params)!=2 or not tunnel_params:
        raise AssertionError("S89 no jointly trainable physical core/tunnel")
    opt=torch.optim.Adam([
        {"params":core_params,"lr":CORE_LR},
        {"params":tunnel_params,"lr":TUNNEL_LR}])
    first,later,pointer=frozen_controllers(old_side)
    rng=random.Random(SEED)
    snapshots=[]
    best=None
    best_key=(-999,-999)
    print(json.dumps({"event":"S89_JOINT_FULL_POLICY_MPS_START",
       "steps":STEPS,"target_wrong_train_count":len(targets),
       "target_ids":[r["id"] for r in targets],
       "protected_interleaved_count":len(protected),
       "trainable_paid_core_codebooks":list(core),
       "trainable_internal_neural_params":sum(p.numel() for p in tunnel_params),
       "QA_or_TEST_imported":False,"LIVE_unchanged":True}),flush=True)
    for step in range(1,STEPS+1):
        opt.zero_grad(set_to_none=True)
        chosen_targets=[
            targets[(2*step-2)%len(targets)],
            targets[(2*step-1)%len(targets)]]
        chosen_protected=rng.sample(protected,2)
        loss_values=[]
        for example,is_old in [(r,False) for r in chosen_targets]+[
                (r,True) for r in chosen_protected]:
            clear(sources)
            # GPU gradients flow through physically paid centroids AND internal
            # rank-8 tunnel; protected examples use old-answer safety margins.
            head=student.lm_head.source().detach()
            loss=policy_loss(student,tok,example,first,later,pointer,head,
                             protected=is_old)
            if not torch.isfinite(loss):
                raise AssertionError("S89 nonfinite training loss")
            loss_values.append(float(loss.detach()))
            (loss/4).backward()
            clear(sources)
        if any(p.grad is None for p in core_params+tunnel_params):
            raise AssertionError("S89 missing GPU gradients on joint neural path")
        core_norm=float(torch.nn.utils.clip_grad_norm_(core_params,1.0))
        tunnel_norm=float(torch.nn.utils.clip_grad_norm_(tunnel_params,1.0))
        if not (core_norm>0 and tunnel_norm>0):
            raise AssertionError("S89 did not train BOTH compressed core and tunnel")
        opt.step()
        with torch.no_grad():
            for q in core_params:q.clamp_(-16,16)
        if step%6==0:
            print(json.dumps({"event":"S89_REAL_MPS_JOINT_STEP",
               "step":step,"loss":sum(loss_values)/len(loss_values),
               "paid_core_gradient_norm":core_norm,
               "internal_tunnel_gradient_norm":tunnel_norm}),flush=True)
        if step in CHECKPOINTS:
            clear(sources)
            raw,side,src_path,side_path,mutation=materialize_candidate(
                old_raw,sources,inner,old_side,a.out,step)
            mini=sample_guard(raw,side,tok,targets,protected,reference)
            within=(len(src_path.read_bytes())+len(side_path.read_bytes())+
                    5000)*8/SOURCE_PARAMS<MAX_BPP
            stats={"step":step,"repacked_source_sha256":hash_bytes(raw),
                "repacked_codebook_cells":mutation["actually_changed_FP16_codebook_cells"],
                "mini":mini,"preflight_0_7_bpp_upper_estimate":within}
            snapshots.append(stats)
            print(json.dumps({"event":"S89_SERIALIZED_CPU_MINI_GUARD",**stats}),flush=True)
            score=(len(mini["gained"])-20*len(mini["lost_full"])-
                   20*len(mini["lost_first"]), len(mini["gained"]))
            if within and score>best_key:
                best_key=score;best=(src_path,side_path,raw,side,mutation,stats)
    inner.uninstall()
    if best is None:
        raise AssertionError("S89 did not produce any physically legal candidate")
    selected_source,selected_side,selected_raw,selected_state,mut,info=best
    out_source=a.out/"S89_UNACCEPTED_SELECTED_CB61.s61"
    out_side=a.out/"S89_UNACCEPTED_SELECTED_JOINT.safetensors"
    out_source.write_bytes(selected_source.read_bytes())
    out_side.write_bytes(selected_side.read_bytes())
    meta={
       "stage":"S89_RESEARCH_ONLY_JOINT_FULL_POLICY_MPS_TRAIN_UNACCEPTED",
       "base_source_S87_sha256":hash_bytes(old_bytes),
       "source_sha256":hash_bytes(selected_raw),
       "source_CB61_physical_sha256":hash_bytes(out_source.read_bytes()),
       "old_S82_sidecar_sha256":hash_bytes(baseline_s82),
       "base_S87_sidecar_sha256":hash_bytes(old_side_raw),
       "prototype_sidecar_sha256":hash_bytes(out_side.read_bytes()),
       "all_previous_inspected_QA_embargoed":True,
       "joint_MPS_train_both_physical_core_and_internal_tunnel":True,
       "S88_zero_gate_CPU_fastpath_maintained":True,
       "S89_selected_STEP":info["step"],
       "targeted_wrong_TRAIN_ids":[r["id"] for r in targets],
       "selected_miniguard":info["mini"],
       "all_snapshot_preflight":snapshots,
       "repacked_mutation":mut,
       "full_456_TRAIN_guard_pending":True,
       "native_C_speed_and_GPU_CPU_parity_pending":True,
       "candidate_accepted":False,
       "LIVE_untouched":True
    }
    data=stable_manifest(meta,out_source.read_bytes(),out_side.read_bytes())
    manifest=a.out/"S89_UNACCEPTED_JOINT_PAID_MANIFEST.json"
    manifest.write_bytes(data)
    print(json.dumps({"event":"S89_UNACCEPTED_JOINT_QUALITY_PROTOTYPE",
        "selected_step":info["step"],
        "new_fully_correct_miniguard":len(info["mini"]["gained"]),
        "lost_full_miniguard":len(info["mini"]["lost_full"]),
        "lost_first_miniguard":len(info["mini"]["lost_first"]),
        "physical_bpp":meta["all_inclusive_physical_bpp"],
        "needs_full_456_guard":True,
        "candidate_accepted":False}),flush=True)


if __name__=="__main__":
    main()
