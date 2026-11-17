"""S22c: 256 genuinely captured Calibri43M+GT TRAIN graphs, gold-free inputs.

Only raw UTF-8 text and native frozen original43M CPU post-GT hidden320
can be passed to build_features. All human event/role/rejected-polarity,
coreference and relation labels are produced in a separate function and
physical file. This stage NEVER optimizes, predicts or scores a model.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from research.nra3_s18.s18c3_raw_russian_grammar_cues import (
    raw_surface_grammar, DIM as RAW_DIM,
)
from research.nra3_s19.s19f_real43m_train9_graph_targets import (
    STATUS, ROLE, ARG_POLARITY, EVENT_RELATION, FrozenModelInputs,
    TrainOnlyTargets,
)
from research.nra3_s19.paid_grammar_parser import INPUT_DIM
from research.nra3_s20.s22_new256_counterfactual_train_graph_source import (
    FROZEN_S22_TRAIN256_SHA256, build_s22_train256,
    text_only_inputs, manifest as source_manifest,
)

STAGE="NRA3_S22_TRUE_ORIGINAL43M_GT_LAYER8_CPU_NEW256_SYNTHETIC_TRAIN_ONLY"
N=256
EXPECTED_COUNTS=(448,976,32,192,192)


def require(ok,what):
    if not bool(ok):
        raise ValueError("S22c genuine 43M/GT TRAIN256 graph: "+what)


@dataclass(frozen=True)
class JoinedSource256:
    inputs: FrozenModelInputs
    gold: TrainOnlyTargets
    graph_sha: str
    original_cpu_hidden_sha: str


def validate_real_capture(feature_file:Path,report_file:Path):
    path,report=Path(feature_file),Path(report_file)
    require(path.is_file() and report.is_file(),"actual original43M hidden and report required")
    m=json.loads(report.read_text(encoding="utf-8"))
    texts=text_only_inputs()
    require(len(texts)==N
            and source_manifest()["source_sha256"]==FROZEN_S22_TRAIN256_SHA256
            and m.get("stage")==STAGE
            and m.get("S22_NEW256_TRAIN_GRAPH_SOURCE_SHA256")==FROZEN_S22_TRAIN256_SHA256
            and m.get("new256_TRAIN_rows")==N
            and m.get("actual_genuine_43M_CPU_new256_TRAIN_prefills")==N
            and m.get("all_616_native_CPU_generated_token_digests_unchanged") is True
            and m.get("original_456_full_EOS")==456
            and m.get("original_new128_full_EOS")==128
            and m.get("GT_residuals_before_readport_in_same_active_S93_scope") is True
            and m.get("actual_43M_CPU_12_transformer_layers") is True
            and m.get("observed_layer_index_zero_based")==7
            and m.get("no_optimizer_or_model_selection") is True
            and m.get("gold_event_roles_coreference_never_passed_to_original43M") is True
            and m.get("no_training_original43M_GT_NRA_or_any_adapter") is True
            and m.get("no_spent_EVAL32_EVAL128_S17c_S15c_S12c") is True
            and m.get("new_candidate_active_616_J7_EOS_not_measured") is True
            and m.get("all_in_bpp_le07_NOT_CERTIFIED") is True
            and m.get("whole_CPU_ge90pct_CB59_NOT_CERTIFIED") is True
            and m.get("NRA3_output_gate")==0
            and m.get("NRA3_candidate_accepted") is False
            and m.get("activation_allowed") is False
            and m.get("actual_fp16_feature_file")==path.name,
            "parent actual original43M CPU/GT/J7/256 source provenance drift")
    physical_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    require(physical_sha==m.get("actual_fp16_feature_sha256"),
            "physical original hidden320 FP16 bytes changed")
    t=load_file(str(path),device="cpu")
    require(set(t)=={"token_hidden320","input_ids","token_mask","token_lengths"},
            "no hidden target/graph allowed in native token features")
    h,ids,mask,length=(t[z] for z in (
        "token_hidden320","input_ids","token_mask","token_lengths"))
    require(h.ndim==3 and h.shape[0]==N and h.shape[-1]==320
            and h.dtype==torch.float16 and ids.dtype==torch.int32
            and mask.dtype==torch.bool and length.dtype==torch.int32
            and length.shape==(N,)
            and ids.shape==mask.shape==h.shape[:2]
            and bool(torch.isfinite(h).all())
            and bool((h[~mask]==0).all())
            and bool((ids[~mask]==-1).all())
            and bool(torch.equal(mask,torch.arange(h.shape[1]).unsqueeze(0)
                                  < length.unsqueeze(1)))
            and length.tolist()==m.get("true_train256_token_counts")
            and list(h.shape)==m.get("actual_hidden320_shape")
            and len(m.get("wordpiece_alignment",()))==N,
            "genuine native BPE padded FP16 original geometry failed")
    rawids=[list(map(int,ids[i,:int(length[i])])) for i in range(N)]
    check=hashlib.sha256(json.dumps(
        rawids,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    require(check==m.get("actual_token_ids_lengths_sha256"),
            "actual original Calibri tokenizer token ids differ")
    for i,text in enumerate(texts):
        row=m["wordpiece_alignment"][i]
        require(row.get("text_sha256")==hashlib.sha256(text.encode()).hexdigest()
                and row.get("token_count")==int(length[i]),
                "original native tokenizer order/word indices differ")
    return texts,h,m,physical_sha


def build_features(texts:tuple[str,...],hidden:torch.Tensor,meta:dict)->FrozenModelInputs:
    """No human graph parameter exists here by design."""
    require(texts==text_only_inputs() and len(texts)==N
            and hidden.ndim==3 and hidden.shape[0]==N
            and hidden.shape[2]==320 and hidden.dtype==torch.float16
            and hidden.device.type=="cpu"
            and len(meta.get("wordpiece_alignment",()))==N,
            "only 256 exact plain TRAIN texts + genuine CPU FP16 embeddings")
    toks=[raw_surface_grammar(s) for s in texts]
    max_w=max(map(len,toks))
    x=torch.zeros(N,max_w,INPUT_DIM,dtype=torch.float32)
    mask=torch.zeros(N,max_w,dtype=torch.bool)
    for i,(sentence,words) in enumerate(zip(texts,toks)):
        m=meta["wordpiece_alignment"][i]
        lasts=m["word_last_token_positions"]
        positions=m["word_token_positions"]
        n=len(words)
        require(n==len(lasts)==len(positions),
                "original native wordpiece count must match plain text")
        occupied=set()
        for j,word in enumerate(words):
            sub=positions[j]
            k=lasts[j]
            require(sentence[word.start:word.end]==word.form
                    and isinstance(sub,list) and len(sub)>0
                    and all(type(z) is int and 0<=z<meta["true_train256_token_counts"][i]
                            for z in sub)
                    and k==max(sub) and not occupied.intersection(sub),
                    "original native token word crossing or human annotation leak")
            occupied.update(sub)
            x[i,j,:320]=hidden[i,k].float()
            x[i,j,320:]=torch.tensor(word.features,dtype=torch.float32)
        mask[i,:n]=True
    require(RAW_DIM==18 and INPUT_DIM==338
            and bool(torch.isfinite(x).all())
            and bool((x[~mask]==0).all())
            and bool(((x[...,320:]==0)|(x[...,320:]==1)).all()),
            "gold-blind input-only tensor ABI drift")
    return FrozenModelInputs(texts,x,mask)


def build_train_targets(data:FrozenModelInputs)->TrainOnlyTargets:
    """Training labels are created ONLY after model inputs are sealed."""
    graphs=build_s22_train256()
    require(data.raw_texts==tuple(g.text for g in graphs)
            and len(graphs)==N,"human template TRAIN256 target order drift")
    valid=data.valid_word_mask
    _,width=valid.shape
    events=torch.full((N,width),-1,dtype=torch.long)
    roles=torch.full((N,width,width),-1,dtype=torch.long)
    polarity=torch.full_like(roles,-1)
    coref=torch.full_like(roles,-1)
    relations=torch.full_like(roles,-1)
    for i,g in enumerate(graphs):
        n=len(g.tokens)
        require(n==int(valid[i].sum()),"word count and raw-only source differ")
        events[i,:n]=0
        roles[i,:n,:n]=0
        coref[i,:n,:n]=0
        relations[i,:n,:n]=0
        for e in g.events:
            require(events[i,e.predicate-1]==0 and e.factuality in STATUS,
                    "duplicate/unsupported event status label")
            events[i,e.predicate-1]=STATUS.index(e.factuality)
        anchors={e.event_id:e.predicate-1 for e in g.events}
        for a in g.arguments:
            p,j=anchors[a.event_id],a.token-1
            require(p!=j and roles[i,p,j]==0
                    and a.role in ROLE and a.polarity in ARG_POLARITY,
                    "duplicate/unsupported predicate-argument polar edge")
            roles[i,p,j]=ROLE.index(a.role)
            polarity[i,p,j]=ARG_POLARITY.index(a.polarity)
        for c in g.coreferences:
            p,j=c.mention-1,c.antecedent-1
            require(p>j and coref[i,p,j]==0,"bad TRAIN-only pronoun arc")
            coref[i,p,j]=1
        for edge in g.event_relations:
            p,j=anchors[edge.source_event],anchors[edge.target_event]
            require(p!=j and relations[i,p,j]==0
                    and edge.relation in EVENT_RELATION,
                    "bad directed event-event TRAIN relation")
            relations[i,p,j]=EVENT_RELATION.index(edge.relation)
        diag=torch.arange(n)
        roles[i,diag,diag]=-1
        coref[i,diag,diag]=-1
        relations[i,diag,diag]=-1
    counts=tuple(int(z) for z in (
        (events>0).sum(),(roles>0).sum(),(polarity==1).sum(),
        (coref==1).sum(),(relations>0).sum()))
    require(counts==EXPECTED_COUNTS,
            "256 synthetic TRAIN graph gold counts differ from sealed source")
    return TrainOnlyTargets(events,roles,polarity,coref,relations)


def join_original43m_new256(feature_file:Path,report_file:Path)->JoinedSource256:
    texts,h,meta,physical_sha=validate_real_capture(feature_file,report_file)
    inputs=build_features(texts,h,meta)
    targets=build_train_targets(inputs)
    return JoinedSource256(inputs,targets,FROZEN_S22_TRAIN256_SHA256,physical_sha)
