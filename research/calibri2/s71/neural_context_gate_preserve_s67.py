"""CALIBRI S71: experimental neural routing over S67 and S69-step4 deltas.

S70 completed operationally but rejected ALL 9 TRAIN interpolation factors:
the one newly correct fresh TRAIN answer at alpha=1.0 loses S67 old
5000_right, while smaller alphas retain old skills but gain no new answer.
This tests a small context-sensitive neural gate g(h) in (0,1) on a
frozen step4 delta, conditioned on FROZEN 36M hidden states.

This is NOT a hand-coded prompt/answer lookup or a test-time QA selector.
The adapter gate is trained on PREEXISTING TRAIN data only. S68 and all
earlier sealed, inspected QA are PERMANENTLY embargoed. All correctness
claims are TRAIN-only, not unseen generalization or FP32 parity.

The S45 source CB59, S61 physical compressed source, S67 first/rank12
and original core/gate/down/old up and the S69 step4 rejected delta
are immutable. A conditional gate adds just 321 trainable parameters
(plus architecture manifest) and is physically included in bpp.
CPU-only and future GPU-independent, no Jessica chat test, no LIVE.
"""
from __future__ import annotations
import argparse, hashlib, json, statistics, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_BYTES,SOURCE_PARAMS,MODEL_SPEC
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s47 import active_small_green_tunnel as s47
from calibri2.s52 import sequence_eos_protected_green_tunnel as s52
from calibri2.s54 import diverse_train_preserve_S52_full_sequences as s54
from calibri2.s56 import paid_first_token_rank2_while_preserving_s54 as s56
from calibri2.s58 import frozen_first_replay_protected_paired_noun as s58
from calibri2.s63.cpu_greedy_from_native_c_restored import make_samples
from calibri2.s65 import train_only_new_rank116_and_guard_old as s65
from calibri2.s69 import train_new_contexts_without_old_skills_loss as s69
from v186.pairs import build_pairs

S67_SHA=s69.CANDIDATE_SHA
SEED=717171
EPOCHS=240
CHECKS=(60,120,180,240)

def sha(raw):return hashlib.sha256(raw).hexdigest()

class NeuralContextMixer(nn.Module):
    """Extra 321 paid weights, not any prompt ID/answer table.

    Return original S67 later correction plus a scaled immutable
    S69 step4 delta based on actual frozen-model context hidden.
    """
    def __init__(self,original:nn.Module,step4_up:torch.Tensor):
        super().__init__()
        self.frozen=original
        self.frozen.eval().requires_grad_(False)
        original_up=original.up.weight.detach().clone()
        difference=step4_up-original_up
        if original_up.shape!=(320,128) or not torch.equal(difference[:,:12],torch.zeros_like(difference[:,:12])):
            raise AssertionError("S71 update must only change 116 new later up columns")
        self.register_buffer("immutable_S69_step4_delta_up",difference)
        self.context_alpha=nn.Linear(320,1,bias=True)
        with torch.no_grad():
            self.context_alpha.weight.zero_()
            self.context_alpha.bias.fill_(-4.0)

    def alpha(self,hidden):
        return torch.sigmoid(self.context_alpha(F.layer_norm(hidden,(320,))))

    def forward(self,hidden):
        frozen_delta=self.frozen(hidden)
        new_delta=F.linear(torch.tanh(self.frozen.down(hidden)),self.immutable_S69_step4_delta_up)
        new_delta=new_delta*torch.sigmoid(self.frozen.gate(hidden))
        return frozen_delta + self.alpha(hidden)*new_delta

def records_for_class(model,sources,tok,pairs,baseline,kind):
    """Only first-gold-correct user-authored TRAIN contexts.

    Protect every originally correct old answer at BOTH noun+EOS; train
    the new gate to select S69 delta only on previously incorrect
    fresh TRAIN examples with already correct first-color token.
    """
    out=[]
    for p in pairs:
        for side in ("left","right"):
            row=p[side]
            ident=f'{p["pair_id"]}_{side}'
            check=baseline[ident]
            actual=check["actual"]
            expected=check["expected"]
            if actual and int(actual[0])==int(expected[0]) and len(expected)==3:
                if kind=="protect":
                    if not check["full_correct"]:continue
                elif kind=="improve":
                    if check["full_correct"]:continue
                else:raise AssertionError("unknown context class")
                for prefix in (expected[:1],expected[:2]):
                    ids=torch.cat((row["prompt_ids"],
                                   torch.tensor(prefix,dtype=torch.long)))
                    hidden,_=s52.frozen_hidden(model,sources,ids)
                    out.append((ident,hidden))
    if not out:raise AssertionError("no label-free TRAIN hidden representations for "+kind)
    return out

def selector_manifest(compressed,controller,out,metrics):
    manifest={
        "stage":"CALIBRI_S71_TRAIN_ONLY_PAID_CONTEXT_NEURAL_GATE",
        "original_S45_source_SHA256":s65.S45_SHA,
        "parent_S67_rank128_controller_SHA256":S67_SHA,
        "physical_S61_CB61_SHA256":sha(compressed),
        "physical_S61_CB61_bytes":len(compressed),
        "combined_controller_rank128_with_context_alpha_SHA256":sha(controller),
        "combined_controller_bytes":len(controller),
        "neural_context_gate_trainable_parameters":321,
        "inference_algorithm":"S67 frozen first rank2; S67 after-first rank128 base + sigmoid(linear(layernorm(hidden)))*frozen_S69_step4_delta_up_after_gate",
        "S69_delta_immutable_and_embedded":True,
        "no_new_CORE_or_source_trainable_weights":True,
        "S68_QA_permanently_embargoed_from_training_and_candidate_selection":True,
        "candidate_only_no_independent_QA_run":True,
        "no_FP32_parity_claim":True,
        "CPU_only_no_GPU_or_Jessica_chat_TEST":True,
        "no_LIVE_Render_Supabase_APK_modified":True,
        "train_protection":metrics,
    }
    for _ in range(32):
        b=json.dumps(manifest,sort_keys=True,indent=2).encode()
        total=len(compressed)+len(controller)+len(b)
        x={"manifest_bytes":len(b),"physical_all_inclusive_bytes":total,
           "physical_all_inclusive_bpp":f"{total*8/SOURCE_PARAMS:.12f}"}
        if all(manifest.get(k)==v for k,v in x.items()):break
        manifest.update(x)
    b=json.dumps(manifest,sort_keys=True,indent=2).encode()
    if manifest["physical_all_inclusive_bytes"]!=len(compressed)+len(controller)+len(b):
        raise AssertionError("S71 total physical bpp failed to converge")
    if manifest["physical_all_inclusive_bytes"]*8/SOURCE_PARAMS>.7:
        raise AssertionError("S71 exceeds fully accounted .7 physical bpp")
    out.write_bytes(b)
    return manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument("S45_CB59_source",type=Path)
    p.add_argument("S67_rank128_controller",type=Path)
    p.add_argument("S67_all_physical_manifest",type=Path)
    p.add_argument("S61_exact_CB61_envelope",type=Path)
    p.add_argument("S69_step4_rejected_resume",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    source=a.S45_CB59_source.read_bytes()
    old_controller=a.S67_rank128_controller.read_bytes()
    compressed=a.S61_exact_CB61_envelope.read_bytes()
    physical=json.loads(a.S67_all_physical_manifest.read_text())
    if (len(source)!=SOURCE_BYTES or sha(source)!=s65.S45_SHA or
        sha(old_controller)!=S67_SHA or
        physical["candidate_S67_rank128_controller_sha256"]!=S67_SHA or
        physical["physical_CB61_source_sha256"]!=sha(compressed) or
        float(physical["all_inclusive_bpp"])>.7):
        raise AssertionError("S71 immutable full physical source mismatch")
    ck=torch.load(str(a.S69_step4_rejected_resume),map_location="cpu",weights_only=True)
    if ck["step"]!=4 or ck["original_frozen_S67_controller_sha256"]!=S67_SHA:
        raise AssertionError("S69 step4 is not the frozen rejected checkpoint")
    weights=load_file(str(a.S67_rank128_controller))
    later_state=ck["later_state"]
    for k in ("down.weight","gate.weight","gate.bias"):
        if not torch.equal(later_state[k],weights["later."+k]):
            raise AssertionError("S69 mutates frozen after-first "+k)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    orig=s30.make_training_pairs(tok)
    aug=s54.build_new_train(tok)
    history=build_pairs(tok,"TRAIN")
    fresh=s69.build_new_train(tok)
    fresh_hash=sha(json.dumps([
        {"pair_id":z["pair_id"],"left":z["left"]["question"],
         "right":z["right"]["question"]}
        for z in fresh],sort_keys=True).encode())
    if (len(orig),len(aug),len(history),len(fresh))!=(32,32,24,16) or ck["fresh_TRAIN_sha256"]!=fresh_hash:
        raise AssertionError("S71 only exact TRAIN examples are permitted")
    model,sources=create_small_paid_model(CB59(source),build_config(MODEL_SPEC))
    model.eval().requires_grad_(False)
    head=model.lm_head.source().detach().cpu().contiguous()
    first=s47.ActiveGreenTunnel(320,2)
    later=s47.ActiveGreenTunnel(320,128)
    first.load_state_dict({k[6:]:v for k,v in weights.items() if k.startswith("first.")})
    later.load_state_dict({k[6:]:v for k,v in weights.items() if k.startswith("later.")})
    first.eval().requires_grad_(False)
    later.eval().requires_grad_(False)
    reference=s69.take_probe(model,sources,first,later,head,tok,orig,aug,history,fresh)
    protected=(
        s58.correct(reference["original"]),s58.correct(reference["augmented"]),
        s58.correct(reference["historical"]),s58.first_correct(reference["original"]),
        s58.first_correct(reference["augmented"]))
    baseq=s69.quality(reference,orig,aug,fresh,protected)
    if tuple(baseq[k] for k in (
        "S30_original_full","S54_augmented_full","historical_full",
        "S30_original_first","S54_augmented_first",
        "old_original_pairs_both","old_augmented_pairs_both",
        "new_fresh_TRAIN_full","new_fresh_TRAIN_first","new_fresh_TRAIN_both_pairs"
    ))!=(42,28,7,58,46,13,7,10,20,1):
        raise AssertionError("S67 accepted skills are not unchanged")
    # All old full correct TRAIN hidden contexts, including fresh correct
    # ones, must NOT receive the aggressive S69 correction.
    old=(
        records_for_class(model,sources,tok,orig,reference["original"],"protect")+
        records_for_class(model,sources,tok,aug,reference["augmented"],"protect")+
        records_for_class(model,sources,tok,history[:4],reference["historical"],"protect")+
        records_for_class(model,sources,tok,fresh,reference["fresh_TRAIN"],"protect")
    )
    novel=records_for_class(model,sources,tok,fresh,reference["fresh_TRAIN"],"improve")
    left=torch.stack([x[1] for x in old])
    right=torch.stack([x[1] for x in novel])
    if left.shape[1]!=320 or right.shape[1]!=320:
        raise AssertionError("context feature topology must be 320")
    mixer=NeuralContextMixer(later,later_state["up.weight"])
    optimizer=torch.optim.AdamW(mixer.context_alpha.parameters(),lr=.045,weight_decay=.00001)
    probes=[]
    selected=None
    best_score=(baseq["new_fresh_TRAIN_full"],baseq["new_fresh_TRAIN_both_pairs"])
    print(json.dumps({"event":"S71_FROZEN_FEATURE_NEURAL_GATE_TRAIN_START",
        "old_correct_context_positions":len(old),
        "fresh_incorrect_first_right_context_positions":len(novel),
        "frozen_immutable_S67":S67_SHA,
        "gate_parameter_count":sum(p.numel() for p in mixer.context_alpha.parameters()),
        "parent_baseline":baseq,
        "qa_used":False}),flush=True)
    for epoch in range(1,EPOCHS+1):
        optimizer.zero_grad(set_to_none=True)
        protected_alpha=mixer.alpha(left)
        novel_alpha=mixer.alpha(right)
        # Balanced BCE. The gating signal only says PRE-EXISTING TRAIN
        # correct or incorrect; there are no evaluation-QA examples.
        loss=(
            F.binary_cross_entropy(protected_alpha,torch.zeros_like(protected_alpha))+
            F.binary_cross_entropy(novel_alpha,torch.ones_like(novel_alpha))
        )
        if not torch.isfinite(loss):raise AssertionError("nonfinite gate loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mixer.context_alpha.parameters(),5.,error_if_nonfinite=True)
        optimizer.step()
        if epoch not in CHECKS:continue
        mixer.eval()
        with torch.no_grad():
            min_new=float(mixer.alpha(right).min())
            max_old=float(mixer.alpha(left).max())
        tested=s69.take_probe(model,sources,first,mixer,head,tok,orig,aug,history,fresh)
        m=s69.quality(tested,orig,aug,fresh,protected)
        fresh_before=s58.correct(reference["fresh_TRAIN"])
        fresh_after=s58.correct(tested["fresh_TRAIN"])
        lost_fresh=sorted(fresh_before-fresh_after)
        now_score=(m["new_fresh_TRAIN_full"],m["new_fresh_TRAIN_both_pairs"])
        preserved=(m["all_previously_correct_S67_identities_preserved"] and
                   m["S67_original_20_opposition_pairs_preserved"] and not lost_fresh and
                   m["new_fresh_TRAIN_first"]>=baseq["new_fresh_TRAIN_first"])
        eligible=(preserved and now_score>best_score and
                  now_score[0]>baseq["new_fresh_TRAIN_full"])
        entry={"epoch":epoch,"gate_BCE_not_generation_quality":float(loss.detach()),
               "old_correct_max_router_alpha":max_old,
               "fresh_wrong_min_router_alpha":min_new,
               "all_previously_correct_full_answers_preserved":preserved,
               "lost_original_fresh_correct_ids":lost_fresh,
               "eligible_TRAIN_only_candidate":eligible,**m}
        probes.append(entry)
        print(json.dumps({"event":"S71_GREEDY_FULL_TRAIN_CONTEXTUAL_GATE_PROBE",**entry}),flush=True)
        if eligible:
            best_score=now_score
            selected={"epoch":epoch,
                      "gate":{k:v.detach().clone() for k,v in mixer.context_alpha.state_dict().items()},
                      "metrics":entry}
    report={
        "stage":"CALIBRI_S71_CONTEXTUAL_NEURAL_GREEN_TUNNEL_GATE_TRAIN_ONLY",
        "S70_full_alpha_line_search_finished_no_eligible_checkpoint":True,
        "S67_original_frozen":S67_SHA,
        "s69_step4_delta_frozen_not_updated":True,
        "train_probes":probes,
        "eligible_preliminary_TRAIN_candidate":selected is not None,
        "S68_independent_QA_permanently_embargoed_from_training_and_selection":True,
        "not_unseen_generalization_or_FP32_parity":True,
        "requires_FRESH_independent_QA_when_future_candidate_passes":True,
        "GPU_used":False,"Jessica_chat_test_run":False,
        "LIVE_modified":False,
    }
    if selected:
        mixer.context_alpha.load_state_dict(selected["gate"])
        samples=make_samples(tok)
        times={}
        for label,tunnel in (("S67",later),("S71",mixer)):
            measurements=[]
            for _,_,q,_ in samples:
                st=time.perf_counter()
                s56.greedy_first_active(model,sources,first,tunnel,head,tok,q)
                measurements.append(time.perf_counter()-st)
            times[label]=statistics.median(measurements)
        ratio=times["S67"]/times["S71"]
        report["actual_CPU_speed_ratio_S71_vs_S67"]=ratio
        report["CPU_speed_90percent_gate_pass"]=ratio>=.9
        if ratio>=.9:
            output=dict(weights)
            output["context_gate.weight"]=selected["gate"]["weight"].detach().clone()
            output["context_gate.bias"]=selected["gate"]["bias"].detach().clone()
            output["context_delta.up.weight"]=mixer.immutable_S69_step4_delta_up.detach().clone()
            combined=a.out/"S71_CONTEXT_NEURAL_PAID_CANDIDATE_ONLY.safetensors"
            save_file(output,str(combined))
            paid=selector_manifest(compressed,combined.read_bytes(),
                a.out/"S71_ALL_BYTES_PAID_NEURAL_GATE_MANIFEST.json",
                selected["metrics"])
            report["checkpoint_SHA256"]=sha(combined.read_bytes())
            report["fully_paid_bpp"]=paid["physical_all_inclusive_bpp"]
            report["chosen_TRAIN_epoch"]=selected["epoch"]
        else:
            report["eligible_preliminary_TRAIN_candidate"]=False
    (a.out/"S71_TRAIN_ONLY_NEURAL_CONTEXT_GATE_RESULT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({"event":"S71_FINAL_TRAIN_ONLY",
        "eligible":report["eligible_preliminary_TRAIN_candidate"],
        "physical_bpp":report.get("fully_paid_bpp"),
        "selected_epoch":report.get("chosen_TRAIN_epoch"),
        "QA_not_run_or_reused":True}),flush=True)

if __name__=="__main__":main()
