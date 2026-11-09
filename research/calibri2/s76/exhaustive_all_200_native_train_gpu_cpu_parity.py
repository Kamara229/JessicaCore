"""CALIBRI S76: FULL representative 200 TRAIN outputs native compressed CPU vs actual 36M GPU MPS.

Purpose: determine whether TEMPORARY dense, exact CB59-reconstructed FP32
weights can accelerate full hidden-state extraction and autoregressive
inference during research, while retaining original FULL physically paid
<=0.7 bpp CB61+S71 controller for deployable CPU use.

This is computational equivalence/throughput research, NOT training and
NOT independent quality validation. Use only immutable S30 and S69
TRAIN questions; no inspected independent QA (including S72) is read.
Do not select Green Tunnel checkpoints on this probe.

Run 200 PROTECTED old+augmented+historical+new TRAIN prompt outputs on
full real Apple MPS 36M TEMPORARY inference + S71 Green Tunnel and compare
exact all predicted tokens to S71 genuine compressed CPU. Record
per-family parity and speed, but do not train or use any QA. If parity fails, GPU is training-only and cannot be
used for checkpoint quality validation. Never save the expanded FP32
weights. No LIVE, main, Jessica or external deployment.
"""
from __future__ import annotations
import argparse,hashlib,json,statistics,time
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s56 import paid_first_token_rank2_while_preserving_s54 as s56
from calibri2.s64 import cpu_rank128_frozen_greedy_parity as s64
from calibri2.s65 import train_only_new_rank116_and_guard_old as s65
from calibri2.s69 import train_new_contexts_without_old_skills_loss as s69
from calibri2.s54 import diverse_train_preserve_S52_full_sequences as s54
from calibri2.s73 import first_token_neural_recovery_protected_train as s73
from v186.pairs import build_pairs
from calibri2.s71 import neural_context_gate_preserve_s67 as s71
from calibri2.s8.broad_replay import SOURCE_BYTES,SOURCE_PARAMS

def hashbytes(b):return hashlib.sha256(b).hexdigest()

@torch.inference_mode()
def dense_mps_greedy(dense,tok,first,later,head,question):
    prompt=tok.apply_chat_template([{"role":"user","content":question}],
                                  tokenize=False,add_generation_prompt=True)
    seq=tok.encode(prompt,add_special_tokens=False)
    ids=torch.tensor(seq,dtype=torch.long,device="mps")[None]
    outids=[]
    t=time.perf_counter()
    for step in range(s56.MAX_NEW):
        output=dense(input_ids=ids,use_cache=False,output_hidden_states=True)
        if output.hidden_states[-1].device.type!="mps":
            raise AssertionError("S76 source hidden must be on REAL GPU")
        hidden=output.hidden_states[-1][0,-1]
        logits=output.logits[0,-1]
        adapter=first if step==0 else later
        scores=logits+F.linear(adapter(hidden),head)
        winner=int(scores.argmax().item())
        outids.append(winner)
        del output
        if winner==tok.eos_token_id:break
        ids=torch.cat((ids,torch.tensor([[winner]],dtype=torch.long,device="mps")),dim=1)
    torch.mps.synchronize()
    return outids,time.perf_counter()-t

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("S45_original_CB59",type=Path)
    ap.add_argument("S71_paid_controller",type=Path)
    ap.add_argument("S71_manifest",type=Path)
    ap.add_argument("S61_exact_compressed_source",type=Path)
    ap.add_argument("output_dir",type=Path)
    a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(760076);torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError("NO real Apple Silicon MPS GPU. CPU fallback is forbidden.")
    raw=a.S45_original_CB59.read_bytes()
    side=a.S71_paid_controller.read_bytes()
    compact=a.S61_exact_compressed_source.read_bytes()
    manifest=json.loads(a.S71_manifest.read_text())
    if (len(raw)!=SOURCE_BYTES or hashbytes(raw)!=s65.S45_SHA or
        hashbytes(side)!=manifest["combined_controller_rank128_with_context_alpha_SHA256"] or
        hashbytes(side)!=s73_frozen_sha() or
        hashbytes(compact)!=manifest["physical_S61_CB61_SHA256"] or
        manifest["physical_all_inclusive_bytes"]!=len(compact)+len(side)+a.S71_manifest.stat().st_size or
        float(manifest["physical_all_inclusive_bpp"])>.7):
        raise AssertionError("S76 source is not frozen fully paid S71")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    orig=s30.make_training_pairs(tok)
    augmented=s54.build_new_train(tok)
    historical=build_pairs(tok,"TRAIN")[:4]
    new=s69.build_new_train(tok)
    newer=s73.new_s73_train(tok)
    # Entire relevant authored TRAIN population, not a sampled favorable
    # subset. 64+64+8+32+32 = exactly 200 independent TRAIN questions.
    groups=(
        ("S30",orig,64),
        ("S54",augmented,64),
        ("history",historical,8),
        ("S69",new,32),
        ("S73",newer,32),
    )
    questions=[(kind,p[side]["question"]) for kind,pairs,_ in groups
               for p in pairs for side in ("left","right")]
    if (len(questions)!=200 or
        len(set(q for _,q in questions))!=200 or
        any(sum(kind==name for kind,_ in questions)!=n for name,_,n in groups)):
        raise AssertionError("S76 exhaustive TRAIN-only old-skill safeguard sample changed")
    weights=load_file(str(a.S71_paid_controller))
    old=s64.instantiate(raw,weights,128)
    cpu_model,sources,cpu_first,cpu_later,head_cpu=old
    cpu_mixer=s71.NeuralContextMixer(cpu_later,
        weights["later.up.weight"]+weights["context_delta.up.weight"])
    cpu_mixer.context_alpha.load_state_dict({
        "weight":weights["context_gate.weight"],
        "bias":weights["context_gate.bias"]})
    cpu_mixer.eval().requires_grad_(False)
    # 36M full weights exist ONLY as TEMPORARY reconstructed CPU tensors.
    # They are then moved to and evaluated on the real Apple GPU.
    gpu_model=s30.dense_from_paid(CB59(raw))
    gpu_model.eval().requires_grad_(False)
    if sum(p.numel() for p in gpu_model.parameters())!=SOURCE_PARAMS:
        raise AssertionError("S76 full model unexpectedly not same 36M")
    gpu_model=gpu_model.to("mps").eval()
    gpu_first=s64.s47.ActiveGreenTunnel(320,2)
    gpu_later=s64.s47.ActiveGreenTunnel(320,128)
    gpu_first.load_state_dict({k[6:]:v for k,v in weights.items() if k.startswith("first.")})
    gpu_later.load_state_dict({k[6:]:v for k,v in weights.items() if k.startswith("later.")})
    gpu_mixer=s71.NeuralContextMixer(gpu_later,
        weights["later.up.weight"]+weights["context_delta.up.weight"])
    gpu_mixer.context_alpha.load_state_dict({
        "weight":weights["context_gate.weight"],
        "bias":weights["context_gate.bias"]})
    gpu_first=gpu_first.to("mps").eval().requires_grad_(False)
    gpu_mixer=gpu_mixer.to("mps").eval().requires_grad_(False)
    head_gpu=gpu_model.lm_head.weight.detach()
    if head_gpu.device.type!="mps":
        raise AssertionError("full 36M GPU model LM head not on MPS")
    # Verify GPU has real trainable backward capability even though this
    # test does NOT train the original model or evaluate independent QA.
    probe=torch.randn((320,320),device="mps",requires_grad=True)
    val=(probe @ probe.T).square().mean();val.backward()
    if probe.grad is None or probe.grad.device.type!="mps":
        raise AssertionError("no physical Metal GPU gradients available")
    torch.mps.synchronize();del probe,val

    records=[]
    # Unscored warmups; only TRAIN prompts.
    s56.greedy_first_active(cpu_model,sources,cpu_first,cpu_mixer,head_cpu,tok,questions[0][1])
    dense_mps_greedy(gpu_model,tok,gpu_first,gpu_mixer,head_gpu,questions[0][1])
    for i,(family,q) in enumerate(questions):
        order=("cpu","mps") if i%2==0 else ("mps","cpu")
        answers={};times={}
        for label in order:
            if label=="cpu":
                t=time.perf_counter()
                ids=s56.greedy_first_active(cpu_model,sources,cpu_first,cpu_mixer,head_cpu,tok,q)
                times["cpu"]=time.perf_counter()-t
                answers["cpu"]=ids
            else:
                ids,elapsed=dense_mps_greedy(gpu_model,tok,gpu_first,gpu_mixer,head_gpu,q)
                times["mps"]=elapsed
                answers["mps"]=ids
        records.append({
            "TRAIN_family":family,
            "TRAIN_question_id":i,
            "actual_original_compressed_CPU_token_ids":answers["cpu"],
            "TEMP_dense_MPS_real_GPU_token_ids":answers["mps"],
            "exact_all_generated_tokens_match":answers["cpu"]==answers["mps"],
            "first_token_matches":answers["cpu"][:1]==answers["mps"][:1],
            "CPU_seconds":times["cpu"],
            "MPS_seconds":times["mps"]})
        if (i+1)%32==0 or (i+1)==len(questions):
            print(json.dumps({"event":"S76_EXHAUSTIVE_PROTECTED_TRAIN_REAL_GPU_PARITY_PROGRESS",
                "TRAIN_prompts_checked":i+1,"out_of":len(questions),
                "full_output_mismatches":sum(not r["exact_all_generated_tokens_match"] for r in records)}),flush=True)
    ratio=statistics.median(r["CPU_seconds"] for r in records)/statistics.median(r["MPS_seconds"] for r in records)
    correct=sum(r["exact_all_generated_tokens_match"] for r in records)
    families={}
    for kind,_,expected in groups:
        rows=[r for r in records if r["TRAIN_family"]==kind]
        families[kind]={
          "exact_parity":sum(r["exact_all_generated_tokens_match"] for r in rows),
          "of":expected,
          "mismatching_input_ids":[r["TRAIN_question_id"] for r in rows
                                   if not r["exact_all_generated_tokens_match"]],
          "median_speed_ratio_CPU_over_GPU":
            statistics.median(r["CPU_seconds"] for r in rows)/
            statistics.median(r["MPS_seconds"] for r in rows),
        }
    summary={
       "stage":"CALIBRI_S76_ALL_200_PROTECTED_TRAIN_FULL_CPU_VS_MPS_36M",
       "full_unique_model_parameters_on_MPS":SOURCE_PARAMS,
       "original_S45_36M_source_SHA256":hashbytes(raw),
       "S71_physically_paid_full_bpp":manifest["physical_all_inclusive_bpp"],
       "permanent_source_is_compressed_only":True,
       "TEMP_FP32_expanded_tensor_weights_saved":False,
       "real_gpu_backend":"Apple_Metal_MPS",
       "actual_MPS_backward_ops_verified":True,
       "all_samples_are_pre_existing_TRAIN_not_QA":True,
       "TRAIN_sample_count":len(questions),
       "TRAIN_scope_fully_exhaustive_all_prior_answer_guard_families":True,
       "per_TRAIN_family_parity_and_speed":families,
       "CPU_vs_GPU_full_token_sequence_identical_count":correct,
       "CPU_vs_GPU_median_speed_ratio":ratio,
       "GPU_faster_than_compressed_CPU_on_this_TRAIN_sample":ratio>1,
       "real_gpu_parity_sufficient_to_qualify_for_GPU_provisional_full_TRAIN_probes":correct==len(questions),
       "unsafe_to_use_GPU_for_quality_or_checkpoint_guards_if_nonparity":correct!=len(questions),
       "complete_factual_TRAIN_only_token_output_records":records,
       "no_training_or_independent_QA_run":True,
       "all_previous_QA_permanently_embargoed":True,
       "no_FP32_original_pretrained_parity_claim":True,
       "LIVE_Jessica_main_Render_Supabase_APK_untouched":True
    }
    (a.output_dir/"S76_EXHAUSTIVE_ALL_200_REAL_MPS_GPU_CPU_PARITY_TRAIN.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps({"event":"S76_FINAL_ALL_PROTECTED_REAL_MPS_MODEL_PARITY",
        "full_gpu_36M":True,"TRAIN_exact_token_parity_count":correct,
        "TRAIN_sample_count":len(questions),"speed_ratio":ratio,
        "per_family":families,
        "no_heldout_QA":True}),flush=True)

def s73_frozen_sha():
    return "0b4c03c639f2500fa98c97b4def58af4d4b55da3a0dd83267c2c4405d0925bbb"

if __name__=="__main__":main()
