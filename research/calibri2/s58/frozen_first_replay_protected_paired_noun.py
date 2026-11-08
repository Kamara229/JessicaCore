"""CALIBRI S58: frozen S56 paid FIRST gate; replay-protected paired NOUN+EOS.

Scientifically declared in advance using TRAIN-only deficits:
S56 already corrected first-token colors on 58/64 original and 46/64
augmented TRAIN questions while only 13/32 and 7/32 opposed TRAIN pairs
were both fully correct. Train only its existing *physically paid* rank12
AFTER-FIRST controller, to exploit preserved first-color gains. S56 paid
rank2 first-token controller, entire 36.38M S45 compressed CB59 host,
and tokenizer are completely frozen. No new parameters or answer tables.
Do NOT load S29 sealed QA, S26 inspected QA, S53, S55, S57 QA
(or even their source modules); no QA prompt/output may influence fitting,
hyperparameters or checkpoint selection. S57 frozen QA aggregate is
not a TRAIN signal. All 32+32 pairs are *previously TRAIN-labeled*
and are explicitly excluded from claims of unseen generalization.

Acceptance precommitted: preserve every original S56 42+27=69 correct
full+EOS answers, all historical 7/8, every original 58+46 correct first
colors, and at least 13+7 both-opposition solved; get >20 two-sided
TRAIN pairs with strict full+EOS. All weights + physical manifest must
fit <=0.7 bpp, and reload same all 64+64+8 native greedy outputs. CPU
sample ratio vs S56 >=0.90 on only four TRAIN questions.
No Jessica chat tests, LIVE/main/Render/Supabase/APK modifications.
"""
from __future__ import annotations
import argparse,hashlib,json,random,statistics,time
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import load_file,save_file
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
from v186.pairs import build_pairs
from v174.run import save

S45_SHA="12a725b07d37a0dae870d7b47666488265a1fa71a58bebf10da1fc7f25a7c398"
S56_SHA="503926839336ed7161786fa6d8ccb2a4ae6fee876cb566d0e52db80ff128f6d4"
S56_MANIFEST_SHA="bb15a2638f9beb33318bbd97e156973043b220e8d8cda81874a69b61798d7c2b"
S56_SIDE_BYTES=39040
S56_MANIFEST_BYTES=704
S56_TOTAL=3181257
S56_BPP="0.699552507279"
RANK_FIRST=2
RANK_LATER=12
SEED=5858
STEPS=240
PROBES=(40,80,120,160,200,240)
LR=.0025
BATCH_PAIRS=12

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def correct(rows):
    return {key for key,r in rows.items() if r["full_correct"]}

def first_correct(rows):
    return {key for key,r in rows.items() if r["first_gold_correct"]}

def both_correct(rows,pairs):
    return sum(rows[f'{p["pair_id"]}_left']["full_correct"] and
               rows[f'{p["pair_id"]}_right']["full_correct"] for p in pairs)

def eligible_real_first_contexts(model,sources,tok,pairs,initial):
    """Only supervised contrast pairs for which frozen S56 first is correct.

    Gold first is checked against actual S56 greedy output; thus no
    first-token teacher forcing or using expected first when incorrect.
    Gold NOUN and its EOS prefix are TRAIN labels, not QA examples.
    """
    included,excluded=[],[]
    for pair in pairs:
        sides={}
        for side in ("left","right"):
            row=pair[side]
            key=f'{pair["pair_id"]}_{side}'
            previous=initial[key]
            if not previous["first_gold_correct"]:
                continue
            first=int(previous["actual"][0])
            if first!=int(row["targets"][0]):
                raise AssertionError("TRAIN first-token correctness changed")
            noun=int(row["targets"][1])
            if int(row["targets"][-1])!=tok.eos_token_id:
                raise AssertionError("real TRAIN target must end with EOS")
            prefix=torch.cat((row["prompt_ids"],torch.tensor([first],dtype=torch.long)))
            noun_h,noun_base=s52.frozen_hidden(model,sources,prefix)
            eos_prefix=torch.cat((prefix,torch.tensor([noun],dtype=torch.long)))
            eos_h,eos_base=s52.frozen_hidden(model,sources,eos_prefix)
            sides[side]={"noun_hidden":noun_h,"noun_base":noun_base,
                         "noun":noun,
                         "eos_hidden":eos_h,"eos_base":eos_base,
                         "eos":int(tok.eos_token_id)}
        if len(sides)==2:
            if sides["left"]["noun"]==sides["right"]["noun"]:
                raise AssertionError("not actually contrasting")
            included.append({"id":pair["pair_id"],**sides})
        else:
            excluded.append(pair["pair_id"])
    return included,excluded

def full_answer_replay(model,sources,tok,pairs,baseline,expected_count):
    """Full native S56 69 originally correct training answers protected
    at their REAL autoregressively generated first/noun and EOS prefixes.
    """
    noun,eos=[],[]
    keep=correct(baseline)
    if len(keep)!=expected_count:
        raise AssertionError("S56 original full count unexpectedly changed")
    for pair in pairs:
        for side in ("left","right"):
            row=pair[side]
            key=f'{pair["pair_id"]}_{side}'
            if key not in keep:continue
            actual=baseline[key]["actual"]
            expected=baseline[key]["expected"]
            if (actual!=expected or len(actual)!=3 or
                int(actual[-1])!=tok.eos_token_id):
                raise AssertionError("only full S56 correct EOS answers can be replayed")
            first,second=int(actual[0]),int(actual[1])
            p1=torch.cat((row["prompt_ids"],torch.tensor([first],dtype=torch.long)))
            p2=torch.cat((p1,torch.tensor([second],dtype=torch.long)))
            nh,nb=s52.frozen_hidden(model,sources,p1)
            eh,eb=s52.frozen_hidden(model,sources,p2)
            noun.append((key,nh,nb,second))
            eos.append((key,eh,eb,int(tok.eos_token_id)))
    def stack(records):
        if len(records)!=expected_count or len(set(x[0] for x in records))!=expected_count:
            raise AssertionError("original full-answer replay identity mismatch")
        return {"hidden":torch.stack([r[1] for r in records]),
                "base":torch.stack([r[2] for r in records]),
                "target":torch.tensor([r[3] for r in records],dtype=torch.long),
                "keys":[r[0] for r in records]}
    return stack(noun),stack(eos),keep

def new_paired_loss(later,head,pairs,rng):
    if not pairs:
        return None
    cases=[pairs[rng.randrange(len(pairs))] for _ in range(BATCH_PAIRS)]
    scores=[]
    for pair in cases:
        l,r=pair["left"],pair["right"]
        x=s47.apply_delta(l["noun_base"],l["noun_hidden"],later,head)
        y=s47.apply_delta(r["noun_base"],r["noun_hidden"],later,head)
        a,b=l["noun"],r["noun"]
        ce=.5*(F.cross_entropy(x[None],torch.tensor([a],dtype=torch.long))+
               F.cross_entropy(y[None],torch.tensor([b],dtype=torch.long)))
        # Pair contrast must beat opposing counterpart, plus full-vocab CE.
        opposite=F.softplus(1.6-(x[a]-x[b]))+F.softplus(1.6-(y[b]-y[a]))
        ex=s47.apply_delta(l["eos_base"],l["eos_hidden"],later,head)
        ey=s47.apply_delta(r["eos_base"],r["eos_hidden"],later,head)
        ee=.5*(F.cross_entropy(ex[None],torch.tensor([l["eos"]],dtype=torch.long))+
                F.cross_entropy(ey[None],torch.tensor([r["eos"]],dtype=torch.long)))
        scores.append(ce+3.0*opposite+.35*ee)
    return torch.stack(scores).mean()

def old_historical_replay(later,head,old_cache):
    c=old_cache["old"]
    logits=s47.apply_delta(c["base"],c["hidden"],later,head)
    return F.cross_entropy(logits,c["target"])

def physical_manifest(source,sidecar,manifest_path):
    weight=sidecar.read_bytes()
    m={"stage":"S58_REPLAY_PROTECTED_PAID_DUAL_GREEN_TUNNEL",
       "source_CB59_sha256":hashlib.sha256(source).hexdigest(),
       "source_CB59_bytes":len(source),
       "parent_S56_dual_controller_sha256":S56_SHA,
       "dual_controller_sha256":hashlib.sha256(weight).hexdigest(),
       "dual_controller_bytes":len(weight),
       "rank_FIRST_frozen":RANK_FIRST,
       "rank_LATER_trained":RANK_LATER,
       "first_adapter_neural_weights_unchanged":True,
       "all_files_accounted_in_bpp":True,
       "no_QA_imports_or_QA_tuning":True}
    converged=False
    for _ in range(50):
        data=json.dumps(m,indent=2,sort_keys=True,ensure_ascii=False).encode()
        total=len(source)+len(weight)+len(data)
        update={"manifest_bytes":len(data),
                "all_inclusive_total_bytes":total,
                "all_inclusive_bpp":f"{8*total/SOURCE_PARAMS:.12f}"}
        if all(m.get(k)==v for k,v in update.items()):
            converged=True
            break
        m.update(update)
    output=json.dumps(m,indent=2,sort_keys=True,ensure_ascii=False).encode()
    if (not converged or total!=m["all_inclusive_total_bytes"]
        or len(source)+len(weight)+len(output)!=total):
        raise AssertionError("physical accounting failed to converge")
    if 8*total/SOURCE_PARAMS>.7:
        raise AssertionError("all-inclusive neural weights plus manifest >0.7bpp")
    manifest_path.write_bytes(output)
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("source",type=Path)
    ap.add_argument("S56_dual",type=Path)
    ap.add_argument("S56_manifest",type=Path)
    ap.add_argument("out",type=Path)
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    rng=random.Random(SEED)
    source=a.source.read_bytes()
    if (len(source)!=SOURCE_BYTES or sha(a.source)!=S45_SHA or
        a.S56_dual.stat().st_size!=S56_SIDE_BYTES or sha(a.S56_dual)!=S56_SHA or
        a.S56_manifest.stat().st_size!=S56_MANIFEST_BYTES or
        sha(a.S56_manifest)!=S56_MANIFEST_SHA):
        raise AssertionError("S56 sealed parent physical artifact mismatch")
    old_manifest=json.loads(a.S56_manifest.read_text(encoding="utf-8"))
    if (old_manifest["all_inclusive_total_bytes"]!=S56_TOTAL or
        old_manifest["all_inclusive_bpp"]!=S56_BPP or
        old_manifest["dual_controller_sha256"]!=S56_SHA or
        old_manifest["source_CB59_sha256"]!=S45_SHA):
        raise AssertionError("S56 immutable physically paid manifest mismatch")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    original=s30.make_training_pairs(tok)
    added=s54.build_new_train(tok)
    old=build_pairs(tok,"TRAIN")
    if len(original)!=32 or len(added)!=32 or len(old)!=24:
        raise AssertionError("training set composition drift")
    model,sources=create_small_paid_model(CB59(source),build_config(MODEL_SPEC))
    model.requires_grad_(False)
    head=model.lm_head.source().detach().cpu().contiguous()
    parameters=load_file(str(a.S56_dual))
    first=s47.ActiveGreenTunnel(320,RANK_FIRST)
    later=s47.ActiveGreenTunnel(320,RANK_LATER)
    first.load_state_dict({k.split(".",1)[1]:v for k,v in parameters.items() if k.startswith("first.")})
    later.load_state_dict({k.split(".",1)[1]:v for k,v in parameters.items() if k.startswith("later.")})
    first.eval().requires_grad_(False)
    frozen_first={k:v.detach().clone() for k,v in first.state_dict().items()}
    later.eval()
    baseline_orig=s56.probe(model,sources,first,later,head,tok,original)
    baseline_add=s56.probe(model,sources,first,later,head,tok,added)
    baseline_old=s56.probe(model,sources,first,later,head,tok,old[:4])
    orig_keep=correct(baseline_orig)
    add_keep=correct(baseline_add)
    old_keep=correct(baseline_old)
    orig_first=first_correct(baseline_orig)
    add_first=first_correct(baseline_add)
    if (len(orig_keep)!=42 or len(add_keep)!=27 or len(old_keep)!=7
        or len(orig_first)!=58 or len(add_first)!=46 or
        both_correct(baseline_orig,original)!=13 or
        both_correct(baseline_add,added)!=7):
        raise AssertionError("S56 exact saved physical full response baseline drift")
    orig_cache,orig_excluded=eligible_real_first_contexts(model,sources,tok,original,baseline_orig)
    added_cache,added_excluded=eligible_real_first_contexts(model,sources,tok,added,baseline_add)
    if not orig_cache or not added_cache:
        raise AssertionError("S56 had no two-sided first-correct TRAIN pairs")
    nounorig,eosorig,old_orig=full_answer_replay(model,sources,tok,original,baseline_orig,42)
    nounadd,eosadd,old_added=full_answer_replay(model,sources,tok,added,baseline_add,27)
    old_cache,old_identity=s47.parse_training_cache(model,sources,tok,original,old)
    if old_identity!=old_keep:
        raise AssertionError("S56 historic seven old skills drifted")
    summary_start={
        "original_64_full":42,"added_64_full":27,"historical_8_full":7,
        "original_32_pairs_both":13,"added_32_pairs_both":7,
        "original_first_correct":58,"added_first_correct":46,
        "eligible_original_pairs":len(orig_cache),"eligible_added_pairs":len(added_cache),
        "excluded_original_wrong_first_pairs":len(orig_excluded),
        "excluded_added_wrong_first_pairs":len(added_excluded)}
    print(json.dumps({"event":"S58_IMMUTABLE_S56_BASELINE",**summary_start},ensure_ascii=False),flush=True)
    # Controller ranks and weight count must be exactly unchanged from S56.
    count_first=sum(p.numel() for p in first.parameters())
    count_later=sum(p.numel() for p in later.parameters())
    if count_first!=1601 or count_later!=8001:
        raise AssertionError("S58 did not keep exact already-paid ranks")
    optimizer=torch.optim.AdamW(later.parameters(),lr=LR,weight_decay=.0001)
    selected=None
    selected_rows=None
    best_pair_count=20
    best_full_count=69
    records=[]
    for step in range(1,STEPS+1):
        later.train()
        optimizer.zero_grad(set_to_none=True)
        left_loss=new_paired_loss(later,head,orig_cache,rng)
        right_loss=new_paired_loss(later,head,added_cache,rng)
        orig_noun,_,_=s52.protected_new_global_winner_replay(later,head,nounorig)
        orig_eos,_,_=s52.protected_new_global_winner_replay(later,head,eosorig)
        add_noun,_,_=s52.protected_new_global_winner_replay(later,head,nounadd)
        add_eos,_,_=s52.protected_new_global_winner_replay(later,head,eosadd)
        old_loss=old_historical_replay(later,head,old_cache)
        loss=left_loss+right_loss+5.0*(orig_noun+add_noun)+6.0*(orig_eos+add_eos)+.8*old_loss
        if not torch.isfinite(loss):
            raise AssertionError("S58 non-finite autograd objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(later.parameters(),10.,error_if_nonfinite=True)
        optimizer.step()
        if step not in PROBES:continue
        later.eval()
        now_orig=s56.probe(model,sources,first,later,head,tok,original)
        now_add=s56.probe(model,sources,first,later,head,tok,added)
        now_old=s56.probe(model,sources,first,later,head,tok,old[:4])
        oc=correct(now_orig);ac=correct(now_add);kc=correct(now_old)
        fc=first_correct(now_orig);gc=first_correct(now_add)
        op=both_correct(now_orig,original);np=both_correct(now_add,added)
        pairs=op+np
        full=len(oc)+len(ac)
        missing_orig=sorted(orig_keep-oc)
        missing_add=sorted(add_keep-ac)
        missing_old=sorted(old_keep-kc)
        missing_first=sorted(orig_first-fc)
        missing_first_add=sorted(add_first-gc)
        preserved=(not missing_orig and not missing_add and not missing_old
                   and not missing_first and not missing_first_add
                   and op>=13 and np>=7)
        improved=(preserved and pairs>best_pair_count and full>=best_full_count)
        if improved:
            best_pair_count=pairs
            best_full_count=full
            selected={k:v.detach().clone() for k,v in later.state_dict().items()}
            selected_rows=(now_orig,now_add,now_old)
        record={
            "step":step,"original_64_full":len(oc),"added_64_full":len(ac),
            "old_8_full":len(kc),"original_both_pairs":op,
            "added_both_pairs":np,"total_both_pairs":pairs,
            "lost_original_42":missing_orig,"lost_added_27":missing_add,
            "lost_old_7":missing_old,"lost_original_58_colors":missing_first,
            "lost_added_46_colors":missing_first_add,
            "accept_train_candidate":improved,"full_train_objective":float(loss.detach()),
            "paired_original_loss":float(left_loss.detach()),
            "paired_added_loss":float(right_loss.detach())}
        records.append(record)
        print(json.dumps({"event":"S58_FROZEN_FIRST_PAIRED_NOUN_TRAIN_PROGRESS",**record},ensure_ascii=False),flush=True)
    saved=False
    manifest=None
    reload_report=None
    cpu=None
    output_side=a.out/"S58_FROZEN_FIRST_PAID_DUAL_GREEN_TUNNEL.safetensors"
    output_manifest=a.out/"S58_ALL_INCLUSIVE_PAID_DUAL_MANIFEST.json"
    if selected is not None:
        later.load_state_dict(selected)
        later.eval()
        tensors={"first."+k:v.detach().cpu().contiguous() for k,v in frozen_first.items()}
        tensors.update({"later."+k:v.detach().cpu().contiguous() for k,v in selected.items()})
        save_file(tensors,str(output_side))
        manifest=physical_manifest(source,output_side,output_manifest)
        loaded=load_file(str(output_side))
        re_first=s47.ActiveGreenTunnel(320,RANK_FIRST)
        re_later=s47.ActiveGreenTunnel(320,RANK_LATER)
        re_first.load_state_dict({k.split(".",1)[1]:v for k,v in loaded.items() if k.startswith("first.")})
        re_later.load_state_dict({k.split(".",1)[1]:v for k,v in loaded.items() if k.startswith("later.")})
        if any(not torch.equal(frozen_first[k],re_first.state_dict()[k]) for k in frozen_first):
            raise AssertionError("S58 physically reloaded frozen first gate was modified")
        re_first.eval()
        re_later.eval()
        check_orig=s56.probe(model,sources,re_first,re_later,head,tok,original)
        check_add=s56.probe(model,sources,re_first,re_later,head,tok,added)
        check_old=s56.probe(model,sources,re_first,re_later,head,tok,old[:4])
        if (check_orig!=selected_rows[0] or check_add!=selected_rows[1]
            or check_old!=selected_rows[2] or
            not orig_keep<=correct(check_orig) or not add_keep<=correct(check_add) or
            not old_keep<=correct(check_old) or
            not orig_first<=first_correct(check_orig) or
            not add_first<=first_correct(check_add) or
            both_correct(check_orig,original)<13 or
            both_correct(check_add,added)<7):
            raise AssertionError("S58 actual saved physical weights lost protected full answer skills")
        checks=[original[i][side]["question"] for i,side in
                ((6,"left"),(16,"right"),(26,"left"),(31,"right"))]
        parent=s47.ActiveGreenTunnel(320,RANK_LATER)
        parent.load_state_dict({k.split(".",1)[1]:v for k,v in parameters.items() if k.startswith("later.")})
        parent.eval()
        timing={}
        for label,adapter in (("S56",parent),("S58",re_later)):
            elapsed=[]
            for q in checks:
                t=time.perf_counter()
                s56.greedy_first_active(model,sources,re_first,adapter,head,tok,q)
                elapsed.append(time.perf_counter()-t)
            timing[label]=elapsed
        speed=statistics.median(timing["S56"])/statistics.median(timing["S58"])
        cpu={"four_TRAIN_questions_wall_seconds":timing,
             "median_speed_ratio_vs_S56":speed,"above_90pct_sample":speed>=.9,
             "not_independent_CPU_throughput_certification":True}
        if speed>=.9:
            saved=True
            reload_report={"all_64_original_free_sequences_identical_after_reload":True,
               "all_64_added_free_sequences_identical_after_reload":True,
               "all_8_old_free_sequences_identical_after_reload":True,
               "original_64_full":len(correct(check_orig)),
               "added_64_full":len(correct(check_add)),
               "old_8_full":len(correct(check_old)),
               "original_32_pairs_both":both_correct(check_orig,original),
               "added_32_pairs_both":both_correct(check_add,added),
               "first_gate_exact_unchanged":True}
        else:
            output_side.unlink()
            output_manifest.unlink()
            manifest=None
    report={"stage":"S58_FROZEN_FIRST_LATER_CAUSAL_REPLAY_TRAIN_ONLY",
        "S45_source_SHA":S45_SHA,"S56_parent_dual_SHA":S56_SHA,
        "source_and_first_controller_immutably_frozen":True,
        "all_experiments_from_original_64_and_added_64_TRAIN_only":True,
        "no_S29_S26_S53_S55_S57_QA_imported_or_scored":True,
        "historical_7_full_answers_guarded":True,
        "full_baseline_69_full_answer_noun_EOS_replay":True,
        "predeclared_steps":STEPS,"predeclared_probes":list(PROBES),
        "baseline":summary_start,"all_training_probes":records,
        "sidecar_accepted_after_reloading_and_CPU":saved,
        "physical_paid_manifest":manifest,
        "physical_native_response_reload":reload_report,
        "CPU_4_TRAIN_pair_ratio":cpu,
        "no_independent_generalization_or_FP32_parity_claim":True,
        "no_Jessica_chat_tests_or_LIVE_changes":True}
    save(a.out/"S58_TRAIN_ONLY_FROZEN_FIRST_PAIRED_NOUN_REPORT.json",report)
    print(json.dumps({"event":"S58_FROZEN_FIRST_PAIRED_NOUN_FINAL",
       "accepted":saved,
       "original_full":reload_report["original_64_full"] if reload_report else 42,
       "added_full":reload_report["added_64_full"] if reload_report else 27,
       "orig_both_pairs":reload_report["original_32_pairs_both"] if reload_report else 13,
       "added_both_pairs":reload_report["added_32_pairs_both"] if reload_report else 7,
       "old7_preserved":True,
       "paid_bpp":manifest["all_inclusive_bpp"] if manifest else None,
       "CPU_sample_ratio":cpu["median_speed_ratio_vs_S56"] if cpu else None,
       "controller_exported":saved},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
