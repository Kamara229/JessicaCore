"""NRA2 S77: causal role separation at admission: EVENT MENTION ≠ asserted EVENT.

Original S52/S57 masks intentionally exclude quoted content from assertive
event candidates. That protects user-facing false transfers, but destroys
the observability of quotes for specialized NRA2 situation micro-areas.
S77 introduces only a NEW read-only MENTION proposal mask; it does not
remove a single old S52/S57/S62/S65 exclusion or change an output token.

All candidate masks and event rankings derive from 120 immutable NEW
TRAIN-only original true Calibri43M+GT H8, original S26/S57/S59/S61/S62/
S63/S65 frozen physical paid modules and literal source text. No gold is
ever used to form a mention candidate or a score. Existing S74 template-
constructed TRAIN-only main-event anchors are unsealed only AFTER every
source-only parent forward for POSTHOC descriptive coverage audit.

This stage is neither an independent validation population nor evidence
of learned mentionhood: source proposal is broad and may include nouns.
Allowing quoted predicates to be considered as MENTIONS is not permission
to mark them asserted, produce recipients, or change original decoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter,defaultdict

import torch
from safetensors.torch import load_file

from research.nra3_s20.s52_paidgraph_native_quote_bpe_abstain import (
    S52QuoteAwareFrozenPaidGraphBridge,tensor_sha,
)
from research.nra3_s20.s57_eventfirst_structured_semantic_decoder import (
    gold_blind_eventfirst_features,NEG,PREPOSITIONS,
)
from research.nra3_s20.s59_native320_contrastive_event_polarity import (
    semantic_features,
)
from research.nra3_s20.s60_native43m_local_triclass_model import (
    source_local_scope,
)
from research.nra3_s20.s74_newtrain120_source_ONLY import (
    new_source_only_train120,source_sha256,GROUP_TRAIN,GROUP_DIAG,
)
from research.nra3_s18.s18c3_raw_russian_grammar_cues import (
    raw_surface_grammar,
)
from research.nra3_s20.s76_train_newtrain120_true43m_8area_factor_router import (
    source_only_original_parent_features,
)
from research.nra3_s20.s76_s74_newtrain120_event_relative_gold_TRAIN_LOSS_ONLY import (
    template_constructed_OLDTRAIN_event_area_gold120_for_loss_only,
)
from research.nra3_s20.s75_original43m_gt_newtrain120_native_h8 import (
    TENSOR_NAME,
)
# No import original S75 CPU tokenizer/model runner, all weights immutable.
STAGE="NRA2_S77_REAL43M_GT_NEWTRAIN120_QUOTE_INCLUSIVE_MENTION_VS_ASSERTED_PROPOSAL_AUDIT"
REPORT="NRA2_S77_TRUE43M_GT_NEWTRAIN120_QUOTED_MENTION_COVERAGE_REPORT.json"


def require(x,reason):
    if not bool(x): raise ValueError("NRA2 S77 mentionhood forensic: "+reason)


def source_only_event_proposals(args):
    """Return source-only masks/ranks before ANY label module is consulted."""
    source,parents,old_hashes,rows=source_only_original_parent_features(args)
    raw=load_file(str(args.s75),device="cpu")
    x=raw["original_real43m_gt_word339_READONLY"]
    valid=raw["original_native_word_valid_mask"]
    unsafe=raw["native_BPE_causally_unsafe_word_mask"]
    n,w,_=x.shape
    require(n==len(rows)==120 and torch.isfinite(x).all()
            and bool(valid[:,0].all()) and not bool((unsafe&~valid).any()),
            "S75 true original H8 and BPE source-only ABI")
    texts=tuple(r["text"] for r in rows)
    old26,old57,old59,*_=parents
    with torch.no_grad():
        obs=S52QuoteAwareFrozenPaidGraphBridge(old26).observe(x,valid,unsafe)
        ev=gold_blind_eventfirst_features(x,obs,texts)
        sem=semantic_features(x,ev)
        fused=source_local_scope(sem,texts)
        oldlogits=tuple(t.detach() for t in old57(ev))
        event_logits=old59(fused.base,oldlogits)[0].detach()
    # S52/S57 assertive event gate: quote words intentionally excluded.
    asserted_candidate=ev.candidate_event
    quote=obs.quote_word_mask
    punct=obs.punctuation_mask
    native_safe=valid & ~unsafe & ~punct
    # Independent MENTION pool. It does not assert any event occurred.
    # Exact bare negation markers and prepositions remain excluded, but
    # quoted verbs are allowed as hypothetical MENTIONS.
    lexical_omit=torch.zeros_like(native_safe)
    for i,text in enumerate(texts):
        words=raw_surface_grammar(text)
        require(len(words)==int(valid[i].sum()),
                "S77 same genuine BPE raw word count")
        for j,t in enumerate(words):
            lexical_omit[i,j]=t.form.casefold() in NEG or t.form.casefold() in PREPOSITIONS
    mention_candidate=native_safe&~lexical_omit
    require(mention_candidate.dtype==torch.bool
            and not bool((mention_candidate&unsafe).any())
            and not bool((mention_candidate&~valid).any())
            and not bool((mention_candidate&punct).any())
            and bool((mention_candidate & quote).any())
            and bool((asserted_candidate & quote).any()) is False,
            "S77 new read-only mention pool must include quotes without unlocking asserted gate")
    # Unmodified S62 S59 frozen score rank, only a new broader pool;
    # topK cannot be oracle-chosen from annotation.
    ranked={}
    for name,mask in (
        ("original_asserted",asserted_candidate),
        ("mention_quote_inclusive",mention_candidate)):
        ranked[name]={}
        for k in (1,4,8,12,16):
            k_eff=min(k,w)
            masked=event_logits.masked_fill(~mask,float("-inf"))
            top=masked.topk(k_eff,dim=-1)
            valid_slots=torch.isfinite(top.values)
            ranked[name][k]=(top.indices.detach(),valid_slots.detach())
    require(all(not t.requires_grad for t in (event_logits,mention_candidate.float())),
            "no student gradients or source/gold scoring attached")
    return {
        "source":source,
        "parents":parents,"old_hashes":old_hashes,
        "texts":rows,
        "valid":valid,"unsafe":unsafe,"quote":quote,
        "punct":punct,"asserted":asserted_candidate,
        "mention":mention_candidate,
        "frozen_event_logits":event_logits,
        "ranked":ranked,
    }


def posthoc_train_only_coverage(obs):
    """No model forward in this function; only TRAIN label comparison."""
    train_gold=template_constructed_OLDTRAIN_event_area_gold120_for_loss_only()
    rows=obs["texts"]
    require(len(train_gold)==len(rows)==120
            and all(g["row_id"]==r["row_id"]
                    for g,r in zip(train_gold,rows)),
            "S77 constructed NEW TRAIN120 gold only after source forward")
    original_sources=obs["source"]["event_indices"]
    original_allowed=obs["source"]["event_allowed"]
    asserted=obs["asserted"];mention=obs["mention"];quote=obs["quote"]
    safe=obs["valid"]&~obs["unsafe"]
    breakdown=defaultdict(lambda:Counter())
    records=[]
    for i,(r,g) in enumerate(zip(rows,train_gold)):
        at=g["main_event_raw_index_ONLY_in_TRAIN_loss"]
        require(0<=at<int(obs["valid"][i].sum()),
                "S77 TRAIN anchor within original word text")
        original_in_beam=bool(((original_sources[i]==at)&original_allowed[i]).any())
        old_mask=bool(asserted[i,at])
        mention_mask=bool(mention[i,at])
        quoted=bool(quote[i,at])
        possible=bool(safe[i,at])
        inside={}
        for name in obs["ranked"]:
            inside[name]={}
            for k,(indices,mask) in obs["ranked"][name].items():
                inside[name][str(k)]=bool(((indices[i]==at)&mask[i]).any())
        original_beam8=inside["original_asserted"]["8"]
        require(original_beam8==original_in_beam,
                "S77 recreated old S65 frozen S62 original 8 source positions drift")
        variant=int(r["row_id"].split("_")[1])
        d=breakdown[f"V{variant:02d}"]
        d["n"]+=1
        d["old_beam8_hits"]+=int(original_in_beam)
        d["old_S57_asserted_mask_eligible"]+=int(old_mask)
        d["quote_inclusive_mention_eligible"]+=int(mention_mask)
        d["gold_target_token_inside_original_quote"]+=int(quoted)
        d["original_native_BPE_valid_nonunsafe"]+=int(possible)
        d["old_S57_mask_loss"]+=int(not old_mask)
        d["old_beam_top8_truncation_after_eligibility"]+=int(old_mask and not original_beam8)
        d["new_mention_top8_hits"]+=int(inside["mention_quote_inclusive"]["8"])
        d["new_mention_top12_hits"]+=int(inside["mention_quote_inclusive"]["12"])
        d["new_mention_top16_hits"]+=int(inside["mention_quote_inclusive"]["16"])
        records.append({
            "raw_source_sha256":hashlib.sha256(r["text"].encode("utf-8")).hexdigest(),
            "row_id_POSTHOC_ONLY":r["row_id"],
            "family_POSTHOC_ONLY":r["family"],
            "variant_POSTHOC_ONLY":variant,
            "constructed_anchor_raw_index_POSTHOC_ONLY":at,
            "original_quoted_word_mask_at_target_POSTHOC":quoted,
            "original_S52_S57_ASSERTED_eligibility_at_target":old_mask,
            "original_S62_S65_beam8_contains_constructed_event":original_in_beam,
            "S77_quote_inclusive_mention_eligibility_at_target":mention_mask,
            "S77_quote_inclusive_mention_top8_contains_constructed_event":
                inside["mention_quote_inclusive"]["8"],
            "S77_quote_inclusive_mention_top12_contains_constructed_event":
                inside["mention_quote_inclusive"]["12"],
            "S77_quote_inclusive_mention_top16_contains_constructed_event":
                inside["mention_quote_inclusive"]["16"],
            "not_actual_recipient_or_assertion_prediction":True,
            "output_gate":0,"candidate_accepted":False,
        })
    old_beam=sum(r["original_S62_S65_beam8_contains_constructed_event"] for r in records)
    mention_mask=sum(r["S77_quote_inclusive_mention_eligibility_at_target"] for r in records)
    mention8=sum(r["S77_quote_inclusive_mention_top8_contains_constructed_event"] for r in records)
    mention12=sum(r["S77_quote_inclusive_mention_top12_contains_constructed_event"] for r in records)
    mention16=sum(r["S77_quote_inclusive_mention_top16_contains_constructed_event"] for r in records)
    original_excluded_quote=sum(
        bool(not z["original_S52_S57_ASSERTED_eligibility_at_target"]
             and z["original_quoted_word_mask_at_target_POSTHOC"])
        for z in records)
    # Broad masks do NOT demonstrate precision. Record candidate
    # inflation so zero false-transfer claims are never inferred.
    n_asserted=int(obs["asserted"].sum())
    n_mentions=int(obs["mention"].sum())
    require(old_beam==96 and sum(not r["original_S62_S65_beam8_contains_constructed_event"] for r in records)==24
            and n_mentions>=n_asserted and len(records)==120
            and all(x["output_gate"]==0 for x in records),
            "S77 original S76 beam count or new mention source changed")
    assert all(z["new_mention_top8_hits"]<=z["new_mention_top12_hits"]<=z["new_mention_top16_hits"]
               for z in breakdown.values())
    return {
        "stage":STAGE,
        "S75_original_43M_GT_native_H8_SHA256":
            "83c1da7b6e79afcfad339c498dcf0f4937411fbbfb081d500d5ca59ea4798900",
        "S74_NEW_TRAIN120_SOURCE_SHA256":source_sha256(),
        "old_S65_32nd_or_held_reuse":False,
        "original_616_J7_protected_baseline_previously_replayed":True,
        "real_original_model_and_old_S26_S57_S59_S61_S62_S63_S65_unchanged":True,
        "S77_mention_masks_computed_source_only_before_train_gold_open":True,
        "original_ASSERTED_gate_remains_quote_EXCLUDING":True,
        "new_MENTION_gate_includes_quoted_text_but_NEVER_asserts":True,
        "NRA2_only_shadow_diagnostic_no_weights_trained":True,
        "n_predeclared_synthetic_NEW_TRAIN":120,
        "n_original_S65_main_event_in_8beam":old_beam,
        "n_original_S65_main_event_MISSING_in_8beam":120-old_beam,
        "n_missing_due_to_quote_mask":original_excluded_quote,
        "n_original_asserted_candidate_tokens":n_asserted,
        "n_new_mention_candidate_tokens":n_mentions,
        "n_new_quote_inclusive_mention_gold_eligible":mention_mask,
        "n_new_quote_inclusive_mention_top8_gold":mention8,
        "n_new_quote_inclusive_mention_top12_gold":mention12,
        "n_new_quote_inclusive_mention_top16_gold":mention16,
        "source_only_topK_has_no_TRAIN_gold_rerank":True,
        "new_mention_pool_has_unmeasured_false_positive_risk":True,
        "not_end_to_end_factuality_or_recipient_improvement":True,
        "no_independent_first_read_or_original_active_616_candidate":True,
        "all_in_bpp_and_CPU_not_measured":True,
        "per_situation_variant_TRAINONLY":{
            k:dict(v) for k,v in sorted(breakdown.items())},
        "per_row_POSTHOC_TRAINONLY":records,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }


def main():
    parser=argparse.ArgumentParser()
    for name in ("s75","s75_report","s26","s57_paid","s59_paid",
                 "s61_paid","s62_paid","s63_paid","s65","s65_manifest","out"):
        parser.add_argument("--"+name.replace("_","-"),required=True,type=Path)
    args=parser.parse_args()
    require(not args.out.exists(),"never overwrite first-read S77 forensic artifact")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    observations=source_only_event_proposals(args)
    result=posthoc_train_only_coverage(observations)
    require(all(tensor_sha(*(p.detach() for p in mod.parameters()))==checksum
                for mod,checksum in zip(
                    observations["parents"],observations["old_hashes"])),
            "S77 frozen original S65 ancestry mutated")
    args.out.mkdir(parents=True,exist_ok=False)
    report=args.out/REPORT
    report.write_text(json.dumps(result,ensure_ascii=False,
                      indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("NRA2_S77_REAL43M_GT_MENTION_VS_ASSERTED_NEWTRAIN120",
          json.dumps({k:result[k] for k in (
              "n_original_S65_main_event_MISSING_in_8beam",
              "n_missing_due_to_quote_mask",
              "n_original_asserted_candidate_tokens",
              "n_new_mention_candidate_tokens",
              "n_new_quote_inclusive_mention_top8_gold",
              "n_new_quote_inclusive_mention_top12_gold",
              "n_new_quote_inclusive_mention_top16_gold",
              "output_gate")},sort_keys=True),flush=True)

if __name__=="__main__":
    main()
