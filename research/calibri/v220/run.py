"""CALIBRI v220: physically paid all-layer activation-alignment Green Tunnel pilot.

Unlike v217-v219 weight SSE fitting or v212 output-only imitation, the
training objective directly aligns original FP32 intermediate layer states
on an immutable public TRAIN corpus. Every step backpropagates to all
211 already-paid 32x8 FP16 codebooks; original 5-bit indices, row scales,
input/output tying, and the native C inference ABI are unchanged.

This is a deliberately exploratory 24-step TRAIN experiment, not a
demonstrated language recovery. No QA/VALID/DEV/TEST decisions enter
training, stopping, layer weights or checkpoint selection. A disjoint
four-record TRAIN-only diagnostic is reported, never a selection gate.
Public DEV16 is repeatedly viewed; report native whole answers only
AFTER physical source is SHA-sealed. No LIVE/main/Jessica modifications.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v169.sources import all_cases
from v174.protocol import freeze_cases
from v174.intervene import extract_original
from v174.run import save
from v176.operators import clear
from v185.codec import replace_discrete
from v186.fit import paired_loss
from v186.run import speed_pairs
from v196 import protocol as p
from v196.data import training_inputs, positions_and_targets
from v196.recover import build_trainable_from_paid_source
from v196.run import newly_frozen_public_dev_16
from v212.run import SOURCE_V206_SHA256

SEED = 220
STEPS = 24
LR = 0.00025
GRAD_CLIP = 1.0
TEMPERATURE = 2.0
# Split is precommitted without inspecting any public QA; every fifth
# index is a TRAIN-only diagnostic, never used for optimizer/stopping.
HELDOUT_TRAIN_INDICES = (4, 9, 14, 19)
TRAIN_INDICES = tuple(i for i in range(20) if i not in HELDOUT_TRAIN_INDICES)
OBJECTIVE_WEIGHTS = (0.55, 0.30, 0.15)  # all-layer state, paired gold, full vocab

def check_protocol():
    assert len(TRAIN_INDICES)==16 and len(HELDOUT_TRAIN_INDICES)==4
    assert set(TRAIN_INDICES).isdisjoint(HELDOUT_TRAIN_INDICES)
    assert set(TRAIN_INDICES)|set(HELDOUT_TRAIN_INDICES)==set(range(20))
    assert STEPS == 24 and abs(sum(OBJECTIVE_WEIGHTS)-1) < 1e-12

@torch.no_grad()
def freeze_original_train_activations(teacher, samples):
    check_protocol()
    if len(samples) != 20:
        raise AssertionError("Original public TRAIN corpus drift")
    out=[]
    teacher.eval()
    for i, sample in enumerate(samples):
        ids=sample["ids"].detach().long().cpu()[None]
        positions,_=positions_and_targets(sample)
        # Precommitted two positions/sample; captures interior language
        # representations and answer contexts rather than just token 1.
        pos=torch.unique(torch.stack((positions[0],positions[-1])))
        o=teacher(input_ids=ids,use_cache=False,output_hidden_states=True)
        states=tuple(h[0,pos].detach().float().clone()
                     for h in o.hidden_states)
        probs=F.softmax(o.logits[0,pos].float()/TEMPERATURE,dim=-1)
        if (len(states)<3 or any(x.shape!=states[0].shape for x in states)
                or probs.shape!=(len(pos),p.VOCAB_SIZE)
                or not torch.isfinite(probs).all()):
            raise AssertionError("Incomplete authentic FP32 layer/vocab reference")
        out.append({"split":"TRAIN","source_index":p.PUBLIC_TRAIN_INDICES[i],
                    "ids":ids,"positions":pos,
                    "layer_states":states,"vocab_probs":probs.to(torch.float16)})
    return out


def feature_objective(student, ref):
    if ref["split"]!="TRAIN":
        raise AssertionError("Cannot optimize on VALID/DEV/TEST")
    output=student(input_ids=ref["ids"],use_cache=False,output_hidden_states=True)
    positions=ref["positions"]
    if len(output.hidden_states)!=len(ref["layer_states"]):
        raise AssertionError("Compressed model lost original FP32 layers")
    layer_terms=[]
    for a,b in zip(output.hidden_states,ref["layer_states"]):
        student_state=a[0,positions].float()
        teacher_state=b
        if student_state.shape!=teacher_state.shape:
            raise AssertionError("Incomplete all-layer hidden alignment")
        # Relative layer scale avoids early/late layer magnitude dominance.
        rms=teacher_state.square().mean(dim=-1,keepdim=True).sqrt().clamp_min(1e-3)
        relative=F.smooth_l1_loss(student_state/rms,teacher_state/rms)
        angle=(1-F.cosine_similarity(student_state,teacher_state,dim=-1)).mean()
        layer_terms.append(0.5*relative+0.5*angle)
    features=torch.stack(layer_terms).mean()
    logits=output.logits[0,positions].float()
    teacher_probs=ref["vocab_probs"].float()
    full_vocab=-(teacher_probs*F.log_softmax(logits/TEMPERATURE,dim=-1)).sum(-1).mean()
    if not torch.isfinite(features+full_vocab):
        raise AssertionError("Nonfinite FP32 feature restoration signal")
    return features,full_vocab


@torch.no_grad()
def measure_train_only(student,sources,refs,indices):
    # Purely descriptive pre- and post-experiment; never selects a step.
    values=[]
    for i in indices:
        clear(sources)
        states,vocab=feature_objective(student,refs[i])
        values.append((float(states),float(vocab)))
    clear(sources)
    return {"records":len(values),"activation_alignment":sum(x[0] for x in values)/len(values),
            "full_vocabulary_teacher_CE_diagnostic":sum(x[1] for x in values)/len(values)}


def fit(student,sources,refs,pairs):
    check_protocol()
    if len(refs)!=20 or len(pairs)!=24 or len(sources)!=211:
        raise AssertionError("TRAIN records, opposing pairs or real source incomplete")
    for i in range(24):
        if pairs[i]["split"]!="TRAIN":
            raise AssertionError("Only original opposing TRAIN pairs permitted")
    pre_train=measure_train_only(student,sources,refs,TRAIN_INDICES)
    pre_hold=measure_train_only(student,sources,refs,HELDOUT_TRAIN_INDICES)
    frozen_codes={n:src.codes.detach().clone() for n,src in sources.items()}
    optimizer=torch.optim.Adam([s.centroids for s in sources.values()],lr=LR)
    records=[]
    for step in range(STEPS):
        clear(sources)
        optimizer.zero_grad(set_to_none=True)
        idx=TRAIN_INDICES[step%len(TRAIN_INDICES)]
        align,soft_vocab=feature_objective(student,refs[idx])
        opposite,paired_stats=paired_loss(student,None,pairs[step])
        total=(OBJECTIVE_WEIGHTS[0]*align+
               OBJECTIVE_WEIGHTS[1]*opposite+
               OBJECTIVE_WEIGHTS[2]*soft_vocab)
        if not torch.isfinite(total):
            raise AssertionError("Nonfinite TRAIN-only Green Tunnel objective")
        total.backward()
        active=sum(s.centroids.grad is not None for s in sources.values())
        if active!=211:
            raise AssertionError("Green Tunnel did not propagate to all 211 physical codebooks")
        norm=float(torch.nn.utils.clip_grad_norm_(
            [s.centroids for s in sources.values()],GRAD_CLIP,
            error_if_nonfinite=True))
        optimizer.step()
        with torch.no_grad():
            for s in sources.values():
                s.centroids.clamp_(-16.0,16.0)
        clear(sources)
        row={"step":step+1,"TRAIN_source_index":p.PUBLIC_TRAIN_INDICES[idx],
             "opposite_TRAIN_pair_id":pairs[step]["pair_id"],
             "all_211_paid_FP16_codebooks_receive_gradient":active,
             "normalized_all_layer_alignment":float(align.detach()),
             "teacher_full_vocab_train_CE":float(soft_vocab.detach()),
             "opposing_full_answer_train_CE":float(opposite.detach()),
             "gradient_norm":norm}
        records.append(row)
        print(json.dumps({"stage":"v220_all_layer_paid_restoration",
            "step":step+1,"all_source_matrices":active,
            "activation_loss":row["normalized_all_layer_alignment"],
            "TRAIN_pair":row["opposite_TRAIN_pair_id"]}),flush=True)
    if any(not torch.equal(sources[n].codes,c) for n,c in frozen_codes.items()):
        raise AssertionError("Inadvertent original 5-bit index mutation")
    post_train=measure_train_only(student,sources,refs,TRAIN_INDICES)
    post_hold=measure_train_only(student,sources,refs,HELDOUT_TRAIN_INDICES)
    return {"completed_paid_steps":STEPS,"train_first":pre_train,"train_final":post_train,
            "disjoint_TRAIN_only_first":pre_hold,"disjoint_TRAIN_only_final":post_hold,
            "precommitted_split":{"trained":list(TRAIN_INDICES),
                                  "diagnostic_only":list(HELDOUT_TRAIN_INDICES)},
            "source_code_groups_unchanged":True,
            "no_QA_VALID_DEV_TEST_optimizer_or_selection":True,
            "steps":records}


def run(v206_path,v161_path,v193_path,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    raw=Path(v206_path).read_bytes()
    if (hashlib.sha256(raw).hexdigest()!=SOURCE_V206_SHA256
            or len(raw)!=p.SOURCE_BYTES):
        raise AssertionError("Not SHA-authentic v206 starting source")
    archive=CB59(raw)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    samples,pairs,_,origin=training_inputs(tok)
    teacher=original_teacher().cpu().eval()
    refs=freeze_original_train_activations(teacher,samples)
    num_fp32_layers=len(refs[0]["layer_states"])
    del teacher
    student,sources=build_trainable_from_paid_source(archive)
    student.eval()
    if (num_fp32_layers!=student.config.num_hidden_layers+1
            or len(sources)!=211):
        raise AssertionError("Missing original all-layer calculation")
    save(out/"PRECOMMITTED_v220_PROTOCOL.json",{
        "experiment":"all_FP32_layer_states_original_public_TRAIN_only",
        "input_v206_sha":SOURCE_V206_SHA256,"steps":STEPS,
        "original_FP32_hidden_state_depth":num_fp32_layers,
        "train_public_indices":list(TRAIN_INDICES),
        "TRAIN_only_diagnostic_indices":list(HELDOUT_TRAIN_INDICES),
        "objective_weights":OBJECTIVE_WEIGHTS,
        "no_DEV_VALID_TEST_or_QA_available_during_fit":True,
        "source_indices_immutable":True,"no_unpaid_inference_weights":True})
    fitted=fit(student,sources,refs,pairs)
    save(out/"ACTUAL_v220_TRAIN_only_24step_all_layer_audit.json",fitted)
    clear(sources)
    packed,seal=replace_discrete(archive,sources,strict=False)
    if (len(packed)!=p.SOURCE_BYTES or seal["complete_bpp"]>0.7
            or seal["sha256"]==SOURCE_V206_SHA256
            or seal["actual_changed_5bit_group_indices"]!=0
            or seal["actually_changed_FP16_codebook_cells"]<=0):
        raise AssertionError("Incorrect paid physical source or no FP16 trained changes")
    target=out/"CALIBRI_v220_all_layer_paid_TRAIN_tunnel.cb59"
    target.write_bytes(packed)
    save(out/"SEALED_v220_source_BEFORE_ANY_public_DEV_QA.json",{
        "sha256":seal["sha256"],"whole_source_bpp":seal["complete_bpp"],
        "FP16_source_cells_changed":seal["actually_changed_FP16_codebook_cells"],
        "5bit_codes_changed":seal["actual_changed_5bit_group_indices"],
        "sealed_without_using_public_DEV_VALID_TEST":True})
    print(json.dumps({"stage":"v220_source_physically_SHA_sealed_before_QA",
        "sha256":seal["sha256"],"bpp":seal["complete_bpp"]}),flush=True)
    del refs
    original=extract_original(Path(v161_path))
    with TemporaryDirectory(dir=out) as folder:
        root=Path(folder)
        for name in ("CB59","CB206","GT220"):(root/name).mkdir()
        (root/"CB59"/"source.cb59").write_bytes(original.blob)
        base,_,_=make_model("baseline",SimpleNamespace(
            original=root/"CB59"/"source.cb59",folder=root/"CB59"))
        old,_,_=make_model("baseline",SimpleNamespace(
            original=Path(v206_path),folder=root/"CB206"))
        new,_,_=make_model("baseline",SimpleNamespace(
            original=target,folder=root/"GT220"))
        if any(v.ndim>=2 for v in new.parameters()):
            raise AssertionError("Unpaid dense inference weights detected")
        qa=newly_frozen_public_dev_16({"CB206":old,"GT220":new},tok,v193_path,out)
        speed=speed_pairs({"CB59":base,"CB186":new},tok,freeze_cases(all_cases()))
    if (qa["scores"]["FP32_existing_135M"]["machine_confirmed"]!=6
            or qa["scores"]["CB206"]["machine_confirmed"]!=3):
        raise AssertionError("Preexisting frozen public DEV rubric drift")
    result={"completed":True,"original_fp32_DEV16":6,
        "sealed_v206_DEV16":3,
        "new_v220_public_DEV16":qa["scores"]["GT220"]["machine_confirmed"],
        "v220_recognized_wrong":qa["scores"]["GT220"]["machine_recognized_wrong"],
        "v220_ungraded":qa["scores"]["GT220"]["ungraded_not_known_wrong"],
        "real_native_CPU_ratio_CB59":speed["median_speed_ratio_vs_CB59"],
        "whole_source_physical_bpp":seal["complete_bpp"],
        "v220_SHA256":seal["sha256"],
        "TRAIN_only_holdout_activation_pre_post":[
            fitted["disjoint_TRAIN_only_first"]["activation_alignment"],
            fitted["disjoint_TRAIN_only_final"]["activation_alignment"]],
        "QA_was_evaluated_after_source_sealed":True,
        "public_DEV16_exploratory_reused_NOT_independent_test":True,
        "never_selected_weights_using_DEV":True,
        "no_Jessica_main_LIVE_Render_Supabase_APK_changes":True}
    save(out/"v220_REAL_native_full_QA_CPU.json",result)
    print(json.dumps({"stage":"v220_real_native_result",**result}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("v206",type=Path)
    parser.add_argument("v161",type=Path)
    parser.add_argument("v193",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    run(args.v206,args.v161,args.v193,args.output)
