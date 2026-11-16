"""S18c4 true original 43M+GT and 43 packed CB59 native C TRAIN96.

Physically train only new 2597 paid FP16 (old 2309 rank8 +
288 raw grammar projection), 300 original TRAIN directed gold links.
All 3288 ordered wordpairs, both orientations and all negatives.

No oracle UD/POS/role enters model; raw text -> 18 indicator bits
aligns with original real tokenizer last-piece word positions.
Train fixed 200 steps, final physically reloaded FP16 only.
No cross validation or independent EVAL in this stage; only TRAIN F1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from torch.nn import functional as F

from direct_codebook_v59 import CB59
from research.nra3_s0.neural_area import NRA3Config
from research.nra3_s2.native_cpu import NRA3NativePackedCPU
from research.nra3_s11.paid_router import PaidThreeWayRouter
from research.nra3_s13.real43m_train72_readouts import S8B_43_CB59_SHA,S11_39_FP16_SHA
from research.nra3_s16.real43m_train72_directed_valence import (
    metrics,
)
from research.nra3_s18.s18c3_raw_russian_grammar_cues import (
    DIM,raw_surface_grammar,
)
from research.nra3_s18.s18c4_paid_rawgrammar_rank8 import (
    PaidRawGrammarDirectedValence, PARAMETERS, PHYSICAL_BYTES,
)
from research.nra3_s18.real43m_train96_physical_source import (
    checked_real_train96, S18B_TRUE_CPU_RUN,
)
from research.nra3_s18.ru96_multiverb_train_source import (
    S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256,
)
from research.nra3_s18.s18c1_original43m_train96_nativeC_rank8 import (
    native_pair_train96, TOTAL_PAIRS, TOTAL_ROLES,
)

STEPS=200
SNAPSHOTS=(0,50,100,150,200)
WEIGHTS=torch.tensor([.20,1.,1.,1.,1.5],dtype=torch.float32)


def require(ok,msg):
    if not bool(ok):
        raise ValueError("S18c4 true43M native43 raw-only grammar TRAIN96: "+msg)


def create_only_raw_text_pair_features(samples,x):
    """No POS/UD, lemma, head, role, labels or gold used to compute features.

    Same all-word-pairs ordered loop as S18c1, but from plain text alone.
    Source TRAIN annotation used solely to *audit* Unicode offset equality,
    never in raw_surface_grammar() or model inputs.
    """
    predicate=[];argument=[];pair_sources=[];ppos=[];apos=[]
    for k,item in enumerate(samples):
        wordspans=raw_surface_grammar(item.row.text)
        # Audit only: original locked 96 row annotation spans must match
        # raw regex spans; annotation itself is NOT model input.
        require(len(wordspans)==len(item.aligned.word_alignment)
                and all((token.start,token.end)==(w.start,w.end)
                        for token,w in zip(wordspans,item.row.words)),
                "original wordpiece-to-RAW plain Unicode offsets fail")
        token_positions=[int(z.last_token) for z in item.aligned.word_alignment]
        raw=[w.features for w in wordspans]
        for i in range(len(raw)):
            for j in range(i+1,len(raw)):
                for predidx,argidx in ((i,j),(j,i)):
                    predicate.append(raw[predidx])
                    argument.append(raw[argidx])
                    ppos.append(token_positions[predidx])
                    apos.append(token_positions[argidx])
                    pair_sources.append(k)
    gp=torch.tensor(predicate,dtype=torch.float32)
    ga=torch.tensor(argument,dtype=torch.float32)
    require(gp.shape==ga.shape==(TOTAL_PAIRS,DIM)
            and pair_sources==x.source_index.tolist()
            and ppos==x.predicate_positions.tolist()
            and apos==x.argument_positions.tolist()
            and bool(((gp==0)|(gp==1)).all())
            and bool(((ga==0)|(ga==1)).all()),
            "unfiltered original Calibri wordpair physical order or raw-source alignment mismatch")
    return gp,ga


def score(model,x,raw_p,raw_a):
    return model.score_pairs(
        x.predicate,x.argument,x.predicate_positions,
        x.argument_positions,x.current_positions,raw_p,raw_a)


def train_full96(x,raw_p,raw_a):
    torch.manual_seed(318040)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    m=PaidRawGrammarDirectedValence().eval()
    require(sum(p.numel() for p in m.parameters())==PARAMETERS==2597,
            "all 2309+288 PAID FP16 parameters must be registered")
    records=[]
    def snapshot(step):
        blob=m.physical_bytes()
        reloaded=PaidRawGrammarDirectedValence.from_physical(blob).eval()
        require(len(blob)==PHYSICAL_BYTES==5286
                and reloaded.physical_bytes()==blob
                and all(not p.requires_grad for p in reloaded.parameters()),
                "physically reloaded complete 2597 FP16 source must be inference-only")
        with torch.inference_mode():
            metric=metrics(score(reloaded,x,raw_p,raw_a),x)
        records.append({
            "step":step,
            "physical_SHA256":hashlib.sha256(blob).hexdigest(),
            "TRAIN_only_exact_directed_role_P_R_F1":metric,
        })
        return blob
    snapshot(0)
    opt=torch.optim.AdamW(m.parameters(),lr=.025,weight_decay=.0002)
    for step in range(1,STEPS+1):
        opt.zero_grad(set_to_none=True)
        logit=score(m,x,raw_p,raw_a)
        ce=F.cross_entropy(logit,x.target,weight=WEIGHTS)
        reg=1e-5*sum(p.square().mean() for p in m.parameters())
        loss=ce+reg
        require(bool(torch.isfinite(loss)), "nonfinite supervised 96 TRAIN loss")
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in m.parameters()),"missing paid grammar or rank8 original gradients")
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
        opt.step()
        if step in SNAPSHOTS[1:]:
            physical=snapshot(step)
    require([z["step"] for z in records]==list(SNAPSHOTS)
            and len(physical)==PHYSICAL_BYTES,
            "fixed full 200 step schedule or FP16 serialization changed")
    return physical,records


def main():
    parser=argparse.ArgumentParser()
    for flag in ("s18b-hidden","s18b-report","s8-cb59","s11-router","out"):
        parser.add_argument("--"+flag,type=Path,required=True)
    args=parser.parse_args()
    require(not args.out.exists(),"never overwrite any original physical train artifacts")
    samples,s18b=checked_real_train96(args.s18b_hidden,args.s18b_report)
    require(len(samples)==96 and S18B_TRUE_CPU_RUN==43910479658
            and s18b["all_616_native_CPU_generated_token_digests_unchanged"]
            and S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256==
               "abde63c34534d90e00a6ac61808d2453ce0cfc966c83d45377c1fb620999cca9",
            "original CPU 96 TRAIN SHA or frozen parent J7 provenance not authenticated")
    b=args.s8_cb59.read_bytes()
    router=args.s11_router.read_bytes()
    require(len(b)==653798 and hashlib.sha256(b).hexdigest()==S8B_43_CB59_SHA
            and len(CB59(b).handles)==43 and CB59(b).meta[
                "original_unique_parameter_count"]==6_106_154,
            "original 43 matrices physical original-C fingerprint drift")
    require(len(router)==123 and hashlib.sha256(router).hexdigest()==S11_39_FP16_SHA
            and PaidThreeWayRouter.from_physical(router).physical_bytes()==router,
            "original 39-param paid selector physical fingerprint drift")
    args.out.mkdir(parents=True,exist_ok=False)
    so=args.out/"s18c4_43_original_packed_CB59_native_C.so"
    subprocess.run(["cc","-O3","-std=c11","-fPIC","-shared",
                    "research/calibri/direct_kernel_v59.c","-o",str(so)],check=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    native=NRA3NativePackedCPU(
        b,NRA3Config(parallel_hypotheses=3,parallel_selector_version=3),
        so,paid_router_physical=router).eval()
    require(native.source_sha256==S8B_43_CB59_SHA
            and native.paid_router.physical_bytes()==router
            and not native.native_cpu_weights_dense_resident
            and native.output_gate==0 and native.candidate_accepted is False
            and all(not p.requires_grad for p in native.parameters()),
            "native CPU original43 CB59 and S11 must stay frozen")
    x,sha=native_pair_train96(samples,native)
    require(x.total_pairs==TOTAL_PAIRS==3288 and x.positive_links==TOTAL_ROLES==300
            and len(sha)==96,"original real43M+GT native original-C all96 full-pair graph missing")
    gp,ga=create_only_raw_text_pair_features(samples,x)
    physical,records=train_full96(x,gp,ga)
    output="NRA3_S18C4_TRUE43M_GT_NATIVE43_S11_NEW96_RAWGRAMMAR_2597FP16.n3s18c4"
    p=args.out/output
    p.write_bytes(physical)
    require(p.read_bytes()==physical and
            PaidRawGrammarDirectedValence.from_physical(physical).physical_bytes()==physical,
            "learned 2597FP16 raw grammar rank8 full paid ABI changed after write")
    final=records[-1]["TRAIN_only_exact_directed_role_P_R_F1"]
    summary={
        "stage":"NRA3_S18C4_TRUE43M_GT_NATIVE43_S11_NEW96_300GOLD_RAW_GRAMMAR_PAID2597FP16_TRAIN_ONLY",
        "S18b_true_frozen43M_GT_96_cpu_run":S18B_TRUE_CPU_RUN,
        "S18b_true_CPU_96_hidden320_FP16_SHA":s18b["actual_fp16_feature_sha256"],
        "original_616_J7_CPU_digest_unchanged_parent":True,
        "S18_new96_full_SOURCE_SHA":S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256,
        "original_43_CB59_C_SHA":S8B_43_CB59_SHA,
        "original_39_FP16_S11_SHA":S11_39_FP16_SHA,
        "real_original_native_C_new96_prefills":96,
        "original_C_sentence_relation_SHA":sha,
        "all_two_way_gold_agnostic_candidate_pairs":x.total_pairs,
        "true_300_directed_gold_links_only_used_for_train_loss":x.positive_links,
        "raw_utf8_text_surface_grammar_DIM":DIM,
        "raw_grammar_no_gold_UPOS_lemma_UD_or_semantic_role_model_input":True,
        "S18c4_original_2309_plus_288_new_FP16_physically_paid":True,
        "S18c4_physical_params":PARAMETERS,
        "S18c4_physical_all_bytes_incl_checksum":PHYSICAL_BYTES,
        "S18c4_full_TRAIN_fixed_optimizer_steps":STEPS,
        "S18c4_reported_checkpoints_no_selection":list(SNAPSHOTS),
        "S18c4_model_filename":output,
        "S18c4_model_SHA256":hashlib.sha256(physical).hexdigest(),
        "S18c4_training_score_only_F1":final["all_two_direction_wordpair_exact_role"]["f1"],
        "S18c4_training_score_only_P":final["all_two_direction_wordpair_exact_role"]["precision"],
        "S18c4_training_score_only_R":final["all_two_direction_wordpair_exact_role"]["recall"],
        "S18c4_training_snapshots":records,
        "no_crossvalidation_or_independent_EVAL_run":True,
        "no_reuse_S17c_S15c_S12c_or_EVAL128":True,
        "original_43M_GT_CB59_S11_all_frozen":True,
        "strict_online_left_to_right_model_word_causality_unproven":True,
        "new_candidate_active_616_J7_EOS_not_measured":True,
        "inclusive_bpp_le_07_NOT_CERTIFIED":True,
        "whole_CPU_ge90pct_CB59_NOT_CERTIFIED":True,
        "research_only":True,"output_gate":0,
        "candidate_accepted":False,"activation_allowed":False,
        "LIVE_Render_Supabase_Jessica_APK_unchanged":True,
    }
    dest=args.out/"NRA3_S18C4_TRUE_ORIGINAL43M_NEW96_PAID_GRAMMAR_TRAIN_REPORT.json"
    dest.write_text(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in
        ("original_C_sentence_relation_SHA","S18c4_training_snapshots")},
        ensure_ascii=False,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
