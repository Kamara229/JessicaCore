"""CALIBRI S70: rescue S69's step-4 TRAIN increment without any old loss.

S69's attempt finished scientific NEGATIVE. At step 4 the new fresh TRAIN
correct answers rose from 10 to 11, but a formerly correct S67 answer
(5000_right) regressed. Other S69 probes worsened; no S69 neural
checkpoint was accepted. This bounded alpha line search starts from
immutable paid S67 and scales only S69 step4's 116 newly trainable
after-first UP columns, preserving old first/rank12/down/gate/CB59.

The locked S68 one-time independent QA is permanently embargoed:
no import, no questions, no scoring, no performance-dependent tuning.
This is solely TRAIN candidate exploration; future genuinely NEW blind
QA required before any claim of generalization or FP32 parity.
CPU-only and resumable, no Jessica chat test, no LIVE.
"""
from __future__ import annotations
import argparse,hashlib,json,statistics,time
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_BYTES, MODEL_SPEC
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s47 import active_small_green_tunnel as s47
from calibri2.s54 import diverse_train_preserve_S52_full_sequences as s54
from calibri2.s56 import paid_first_token_rank2_while_preserving_s54 as s56
from calibri2.s58 import frozen_first_replay_protected_paired_noun as s58
from calibri2.s63.cpu_greedy_from_native_c_restored import make_samples
from calibri2.s65 import train_only_new_rank116_and_guard_old as s65
from calibri2.s69 import train_new_contexts_without_old_skills_loss as s69
from v186.pairs import build_pairs

ALPHAS=(0.0625,0.125,0.1875,0.25,0.375,0.5,0.625,0.75,1.)
S67_SHA=s69.CANDIDATE_SHA
FIXED_FIRST_RANK=2
FIXED_LATER_OLD_RANK=12

def sha(raw):return hashlib.sha256(raw).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("S45_immutable_CB59",type=Path)
    p.add_argument("S67_immutable_paid_rank128",type=Path)
    p.add_argument("S67_manifest",type=Path)
    p.add_argument("S61_exact_compressed_source",type=Path)
    p.add_argument("S69_negative_step4_resume",type=Path)
    p.add_argument("output_dir",type=Path)
    a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(700070)
    torch.set_num_threads(2)
    raw=a.S45_immutable_CB59.read_bytes()
    frozen=a.S67_immutable_paid_rank128.read_bytes()
    compressed=a.S61_exact_compressed_source.read_bytes()
    physical=json.loads(a.S67_manifest.read_text())
    if (len(raw)!=SOURCE_BYTES or sha(raw)!=s65.S45_SHA or
        sha(frozen)!=S67_SHA or
        physical["candidate_S67_rank128_controller_sha256"]!=S67_SHA or
        physical["physical_CB61_source_sha256"]!=sha(compressed) or
        float(physical["all_inclusive_bpp"])>.7):
        raise AssertionError("S70 source/S67 are not the frozen physically paid artifacts")
    ck=torch.load(str(a.S69_negative_step4_resume),map_location="cpu",weights_only=True)
    if ck.get("step")!=4 or ck.get("original_frozen_S67_controller_sha256")!=S67_SHA:
        raise AssertionError("S69 step4 optimizer checkpoint not exact S67 provenance")
    torch.manual_seed(700070)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    orig=s30.make_training_pairs(tok)
    aug=s54.build_new_train(tok)
    history=build_pairs(tok,"TRAIN")
    fresh=s69.build_new_train(tok)
    fresh_hash=sha(json.dumps([
        {"pair_id":r["pair_id"],"left":r["left"]["question"],"right":r["right"]["question"]}
        for r in fresh],sort_keys=True).encode())
    if (len(orig),len(aug),len(history),len(fresh))!=(32,32,24,16) or ck.get("fresh_TRAIN_sha256")!=fresh_hash:
        raise AssertionError("old/fresh TRAIN input data has drifted")
    weights=load_file(str(a.S67_immutable_paid_rank128))
    original_up=weights["later.up.weight"].detach().clone()
    step4=ck["later_state"]
    for k in ("down.weight","gate.weight","gate.bias"):
        if not torch.equal(step4[k],weights["later."+k]):
            raise AssertionError("S69 improperly mutated frozen later "+k)
    new_up=step4["up.weight"].detach()
    if original_up.shape!=(320,128) or new_up.shape!=original_up.shape:
        raise AssertionError("unexpected Green Tunnel weight topology")
    if not torch.equal(new_up[:,:12],original_up[:,:12]):
        raise AssertionError("S69 step4 lost immutable S56 after-first rank12")
    delta=new_up-original_up
    if torch.count_nonzero(delta[:,12:]).item()==0 or torch.count_nonzero(delta[:,:12]).item()!=0:
        raise AssertionError("S69 step4 has no properly isolated new neural weights")
    model,sources=create_small_paid_model(CB59(raw),build_config(MODEL_SPEC))
    model.eval().requires_grad_(False)
    head=model.lm_head.source().detach().cpu().contiguous()
    first=s47.ActiveGreenTunnel(320,2)
    later=s47.ActiveGreenTunnel(320,128)
    first.load_state_dict({k[6:]:v for k,v in weights.items() if k.startswith("first.")})
    later.load_state_dict({k[6:]:v for k,v in weights.items() if k.startswith("later.")})
    first.eval().requires_grad_(False)
    later.eval().requires_grad_(False)
    base=s69.take_probe(model,sources,first,later,head,tok,orig,aug,history,fresh)
    all_protected=(
        s58.correct(base["original"]),s58.correct(base["augmented"]),
        s58.correct(base["historical"]),s58.first_correct(base["original"]),
        s58.first_correct(base["augmented"])
    )
    fresh_protected=(s58.correct(base["fresh_TRAIN"]),s58.first_correct(base["fresh_TRAIN"]))
    anchor=s69.quality(base,orig,aug,fresh,all_protected)
    if (
        anchor["S30_original_full"],anchor["S54_augmented_full"],anchor["historical_full"],
        anchor["S30_original_first"],anchor["S54_augmented_first"],
        anchor["old_original_pairs_both"],anchor["old_augmented_pairs_both"],
        anchor["new_fresh_TRAIN_full"],anchor["new_fresh_TRAIN_first"],
        anchor["new_fresh_TRAIN_both_pairs"]
    )!=(42,28,7,58,46,13,7,10,20,1):
        raise AssertionError("S70 baseline S67 TRAIN skill/previous correct identities drift")
    probes=[]
    candidate=None
    key_to_beat=(anchor["new_fresh_TRAIN_full"],anchor["new_fresh_TRAIN_both_pairs"])
    for alpha in ALPHAS:
        with torch.no_grad():later.up.weight.copy_(original_up+float(alpha)*delta)
        before=time.perf_counter()
        rows=s69.take_probe(model,sources,first,later,head,tok,orig,aug,history,fresh)
        q=s69.quality(rows,orig,aug,fresh,all_protected)
        new_full=s58.correct(rows["fresh_TRAIN"])
        new_first=s58.first_correct(rows["fresh_TRAIN"])
        old_fresh_full_lost=sorted(fresh_protected[0]-new_full)
        old_fresh_first_lost=sorted(fresh_protected[1]-new_first)
        safe=(q["all_previously_correct_S67_identities_preserved"] and
              q["S67_original_20_opposition_pairs_preserved"] and
              not old_fresh_full_lost and not old_fresh_first_lost and
              q["new_fresh_TRAIN_first"]>=anchor["new_fresh_TRAIN_first"])
        score=(q["new_fresh_TRAIN_full"],q["new_fresh_TRAIN_both_pairs"])
        accepted=safe and score>key_to_beat and score[0]>anchor["new_fresh_TRAIN_full"]
        report={"S69_step4_update_fraction":alpha,"actual_full_greedy_TRAIN_probe_seconds":time.perf_counter()-before,
                "all_S67_and_fresh_previously_correct_identities_preserved":safe,
                "fresh_old_correct_lost":old_fresh_full_lost,"fresh_old_first_lost":old_fresh_first_lost,
                "eligible_preliminary_TRAIN_only_candidate":accepted,**q}
        probes.append(report)
        print(json.dumps({"event":"S70_TRUE_NATIVE_NEW_TRAIN_FROZEN_SKILL_LINE_SEARCH",**report}),flush=True)
        if accepted:
            key_to_beat=score
            candidate={"alpha":alpha,"up":later.up.weight.detach().clone(),"report":report}
    result={
        "stage":"CALIBRI_S70_S69_NEGATIVE_STEP4_BOUNDED_TRAIN_LINE_SEARCH",
        "S69_all_3_greedy_probes_rejected_due_to_lost_S67_full_answers":True,
        "first_stage_S67_reference":anchor,
        "S69_step4_up_only_actual_native_TRAIN_probes":probes,
        "S68_independent_QA_is_inspected_and_PERMANENTLY_EMBARGOED":True,
        "no_QA_imported_no_QA_guided_hyperparam_or_model_selection":True,
        "genuine_NEW_QA_and_real_CPU_speed_required_before_promotion":True,
        "GPU_used":False,"Jessica_chat_test_run":False,"LIVE_modified":False,
        "selected_preliminary_TRAIN_candidate":candidate is not None,
    }
    if candidate is not None:
        with torch.no_grad():later.up.weight.copy_(candidate["up"])
        samples=make_samples(tok)
        medians={}
        for which,w in (("original",original_up),("candidate",candidate["up"])):
            with torch.no_grad():later.up.weight.copy_(w)
            times=[]
            for _,_,question,_ in samples:
                start=time.perf_counter()
                s56.greedy_first_active(model,sources,first,later,head,tok,question)
                times.append(time.perf_counter()-start)
            medians[which]=statistics.median(times)
        ratio=medians["original"]/medians["candidate"]
        result["cpu_speed_vs_S67"]=ratio
        result["CPU_speed_90percent_gate_pass"]=ratio>=0.9
        if ratio>=.9:
            with torch.no_grad():later.up.weight.copy_(candidate["up"])
            path=a.output_dir/"S70_PRELIMINARY_TRAIN_ONLY_GREEN_TUNNEL.safetensors"
            s65.sidecar(first,later,path)
            m=s65.saved_physical_manifest(compressed,path.read_bytes(),
                    a.output_dir/"S70_FULLY_PAID_TRAIN_MANIFEST.json")
            result["selected_ALPHA"]=candidate["alpha"]
            result["all_inclusive_bpp"]=m["all_inclusive_bpp"]
            result["candidate_controller_sha256"]=sha(path.read_bytes())
        else:
            result["selected_preliminary_TRAIN_candidate"]=False
    (a.output_dir/"S70_LINE_SEARCH_ACTUAL_TRAIN_RESULTS.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({"event":"S70_FINAL",
       "preliminary_TRAIN_candidate":result["selected_preliminary_TRAIN_candidate"],
       "all_inclusive_bpp":result.get("all_inclusive_bpp"),
       "best_alpha":result.get("selected_ALPHA"),
       "no_generalization_or_FP32_parity_claim":True}),flush=True)

if __name__=="__main__":
    main()
