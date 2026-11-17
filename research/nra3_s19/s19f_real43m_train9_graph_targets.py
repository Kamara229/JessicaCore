"""S19f: bind REAL frozen Calibri43M+Green Tunnel new9 CPU TRAIN hidden to graphs.

Strictly *data plumbing*, no gradient, no test reuse, no new scores.
The neural feature function accepts plain TEXT, exact original BPE/hidden
and deterministic raw18 text cues; it cannot receive gold graphs.
Gold event, directed role/polarity, coreference and event links are kept
in a separate typed TRAIN target bundle, never attached to the input tensor.
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
from research.nra3_s19.graph_schema import raw_model_input
from research.nra3_s19.train_only_event_graph_source import (
    build_s19_train_only, TRAIN_ONLY_FROZEN_SHA256, train_only_manifest,
)
from research.nra3_s19.s19e_original43m_raw_alignment import s19_train_texts
from research.nra3_s19.paid_grammar_parser import INPUT_DIM

S19E_STAGE = "NRA3_S19E_TRUE_ORIGINAL43M_GT_LAYER8_CPU_NEW9_TRAIN_ONLY"
STATUS = ("NONE", "ASSERTED", "NEGATED", "FUTURE_UNREALIZED", "CONDITIONAL", "UNKNOWN")
ROLE = ("NONE", "AGENT", "PATIENT", "RECIPIENT", "SOURCE", "DESTINATION",
        "EXPERIENCER", "INSTRUMENT", "LOCATION")
ARG_POLARITY = ("POSITIVE", "REJECTED")
EVENT_RELATION = ("NONE", "COORDINATION", "CONDITION_RESULT",
                  "TEMPORAL_SEQUENCE", "CONTRAST")


def require(ok, msg):
    if not bool(ok):
        raise ValueError("S19f genuine original43M+GT NEW9 TRAIN graph: " + msg)


@dataclass(frozen=True)
class FrozenModelInputs:
    """No gold/parse/predicate/role or event-truth fields can appear here."""
    raw_texts: tuple[str, ...]
    raw_word_hidden320_and_raw18: torch.Tensor  # [9,W,338]
    valid_word_mask: torch.Tensor                 # [9,W], right-padded


@dataclass(frozen=True)
class TrainOnlyTargets:
    """Human gold for future optimizer losses only: not a model feature."""
    event_status: torch.Tensor                    # [9,W], -1 padding, 0=NONE
    predicate_argument_role: torch.Tensor          # [9,W,W], -1 illegal
    argument_polarity: torch.Tensor                 # [9,W,W], -1 non-edge
    coreference: torch.Tensor                      # [9,W,W] mention->antecedent
    event_links: torch.Tensor                      # [9,W,W] predicate->predicate


@dataclass(frozen=True)
class JoinedGraphTrain9:
    inputs: FrozenModelInputs
    targets: TrainOnlyTargets
    graph_sha: str
    physical_hidden_sha: str
    original_j7_parent_616_proven: bool


def load_real_new9_feature_tensors(feature: Path, report: Path):
    feature, report = Path(feature), Path(report)
    require(feature.is_file() and report.is_file(),
            "physical true-43M CPU hidden and JSON source manifest required")
    meta = json.loads(report.read_text(encoding="utf-8"))
    texts = s19_train_texts()
    require(
        meta.get("stage") == S19E_STAGE
        and meta.get("S19_NEW9_TRAIN_GRAPH_SOURCE_SHA256")
             == TRAIN_ONLY_FROZEN_SHA256
        and meta.get("new9_TRAIN_rows") == 9
        and meta.get("actual_genuine_43M_CPU_new9_TRAIN_prefills") == 9
        and meta.get("all_616_native_CPU_generated_token_digests_unchanged") is True
        and meta.get("gold_event_roles_coreference_never_passed_to_original43M") is True
        and meta.get("no_training_original43M_GT_NRA_or_any_adapter") is True
        and meta.get("no_optimizer_or_model_selection") is True
        and meta.get("no_spent_EVAL32_EVAL128_S17c_S15c_S12c") is True
        and meta.get("original_456_full_EOS") == 456
        and meta.get("original_new128_full_EOS") == 128
        and meta.get("observed_layer_index_zero_based") == 7
        and meta.get("GT_residuals_before_readport_in_same_active_S93_scope") is True
        and meta.get("true_real_Calibri_token_alignment_all_9") is True
        and meta.get("NRA3_output_gate") == 0
        and meta.get("NRA3_candidate_accepted") is False
        and meta.get("activation_allowed") is False
        and meta.get("new_candidate_active_616_J7_EOS_not_measured") is True
        and meta.get("all_in_bpp_le07_NOT_CERTIFIED") is True
        and meta.get("whole_CPU_ge90pct_CB59_NOT_CERTIFIED") is True
        and meta.get("actual_fp16_feature_file") == feature.name,
        "source SHA/original real CPU J7/no activation provenance drift")
    physical_sha = hashlib.sha256(feature.read_bytes()).hexdigest()
    require(meta.get("actual_fp16_feature_sha256") == physical_sha,
            "physical frozen post-GT layer8 feature SHA drift")
    tensors = load_file(str(feature),device="cpu")
    require(set(tensors) == {"token_hidden320","input_ids","token_mask","token_lengths"},
            "only original native CPU hidden and original tokenizer input IDs allowed")
    h,ids,mask,lens = (tensors[k] for k in (
        "token_hidden320","input_ids","token_mask","token_lengths"))
    require(
        h.ndim == 3 and h.shape[0] == 9 and h.shape[2] == 320
        and h.dtype == torch.float16
        and mask.dtype == torch.bool and ids.dtype == torch.int32
        and lens.dtype == torch.int32 and lens.shape == (9,)
        and h.shape[:2] == ids.shape == mask.shape
        and bool(torch.isfinite(h).all())
        and bool(torch.equal(mask,torch.arange(h.shape[1]).unsqueeze(0)
                    < lens.unsqueeze(1)))
        and bool((h[~mask] == 0).all())
        and bool((ids[~mask] == -1).all()),
        "original tensor shape/FP16/native IDs/right-padding invalid")
    require(lens.tolist() == meta.get("true_train9_token_counts")
            and list(h.shape) == meta.get("actual_hidden320_shape")
            and len(meta.get("wordpiece_alignment",[])) == 9,
            "original native BPE map report and physical padded tokens drift")
    expected_ids = [list(int(t) for t in ids[i,:int(lens[i])]) for i in range(9)]
    actual_id_sha = hashlib.sha256(json.dumps(
        expected_ids,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    require(meta.get("actual_token_ids_lengths_sha256") == actual_id_sha,
            "original 43M token IDs do not match SHA-locked source")
    for i,text in enumerate(texts):
        row = meta["wordpiece_alignment"][i]
        require(row.get("text_sha256")==
                hashlib.sha256(text.encode("utf-8")).hexdigest()
                and row.get("token_count")==int(lens[i]),
                "original tokenizer prompt SHA/order changed")
    return tensors,meta,physical_sha


def assemble_gold_blind_338(texts, h, report):
    """No human annotations accepted. Only exact raw text + frozen hidden."""
    require(isinstance(texts,tuple) and texts==s19_train_texts(),
            "only unchanged preregistered TRAIN9 plaintext allowed")
    require(h.ndim==3 and h.shape[0]==len(texts) and h.shape[2]==320
            and h.dtype==torch.float16 and h.device.type=="cpu"
            and len(report.get("wordpiece_alignment",()))==len(texts),
            "genuine CPU token hidden + original BPE mapping only")
    raw_all=[raw_surface_grammar(t) for t in texts]
    max_w=max(len(tokens) for tokens in raw_all)
    inputs=torch.zeros((len(texts),max_w,INPUT_DIM),dtype=torch.float32)
    valid=torch.zeros((len(texts),max_w),dtype=torch.bool)
    for i,(text,raw) in enumerate(zip(texts,raw_all)):
        bp=report["wordpiece_alignment"][i]
        last=bp.get("word_last_token_positions")
        positions=bp.get("word_token_positions")
        n=len(raw)
        require(len(last)==len(positions)==n,
                "native tokenizer span count must equal raw word count")
        occupied=set()
        for j,tok in enumerate(raw):
            require(text[tok.start:tok.end]==tok.form,
                    "raw source Unicode offsets must not be gold anchors")
            pieces=positions[j]
            k=last[j]
            require(isinstance(pieces,list) and len(pieces)>0
                    and all(type(v) is int and 0<=v<report["true_train9_token_counts"][i]
                            for v in pieces)
                    and k==max(pieces)
                    and not occupied.intersection(pieces),
                    "S19e original native BPE word alignment crosses words")
            occupied.update(pieces)
            inputs[i,j,:320]=h[i,k].float()
            inputs[i,j,320:]=torch.tensor(tok.features,dtype=torch.float32)
        valid[i,:n]=True
        require(bool(torch.isfinite(inputs[i,:n]).all())
                and bool(((inputs[i,:n,320:]==0.)|
                         (inputs[i,:n,320:]==1.)).all()),
                "original 43M hidden/raw-only surface ABI corrupt")
    require(INPUT_DIM==338 and RAW_DIM==18 and valid.sum()>0
            and bool((inputs[~valid]==0.).all()),
            "gold-blind padded word features invalid")
    return FrozenModelInputs(texts,inputs,valid)


def make_train9_targets(input_data: FrozenModelInputs):
    """Here—and ONLY here—access the human TRAIN graph annotations."""
    graphs=build_s19_train_only()
    require(input_data.raw_texts==tuple(g.text for g in graphs)
            and len(graphs)==9,
            "source TRAIN graph and plain-text tensor order mismatch")
    valid=input_data.valid_word_mask
    num,width=valid.shape
    ev=torch.full((num,width),-1,dtype=torch.long)
    role=torch.full((num,width,width),-1,dtype=torch.long)
    polar=torch.full_like(role,-1)
    coref=torch.full_like(role,-1)
    links=torch.full_like(role,-1)
    for i,g in enumerate(graphs):
        n=len(g.tokens)
        require(int(valid[i].sum())==n,
                "human graph word indices must match raw unannotated text")
        ev[i,:n]=0
        role[i,:n,:n]=0
        coref[i,:n,:n]=0
        links[i,:n,:n]=0
        # None of the four targets below ever enters assemble_gold_blind_338.
        for e in g.events:
            require(ev[i,e.predicate-1]==0 and e.factuality in STATUS,
                    "ambiguous or invalid event factuality TRAIN label")
            ev[i,e.predicate-1]=STATUS.index(e.factuality)
        for a in g.arguments:
            eid=next(e.predicate for e in g.events if e.event_id==a.event_id)
            p,j=eid-1,a.token-1
            require(p!=j and role[i,p,j]==0 and a.role in ROLE
                    and a.polarity in ARG_POLARITY,
                    "ambiguous TRAIN directed role/polarity target")
            role[i,p,j]=ROLE.index(a.role)
            polar[i,p,j]=ARG_POLARITY.index(a.polarity)
        for c in g.coreferences:
            p,j=c.mention-1,c.antecedent-1
            require(p>j and coref[i,p,j]==0,
                    "invalid or duplicate coreference target")
            coref[i,p,j]=1
        for rel in g.event_relations:
            p=next(e.predicate for e in g.events
                   if e.event_id==rel.source_event)-1
            j=next(e.predicate for e in g.events
                   if e.event_id==rel.target_event)-1
            require(p!=j and links[i,p,j]==0 and rel.relation in EVENT_RELATION,
                    "duplicate or unsupported directed event relation")
            links[i,p,j]=EVENT_RELATION.index(rel.relation)
        diagonal=torch.arange(n)
        role[i,diagonal,diagonal]=-1
        coref[i,diagonal,diagonal]=-1
        links[i,diagonal,diagonal]=-1
    require(
        int((ev>0).sum())==15 and int((role>0).sum())==40
        and int((polar==1).sum())==1 and int((coref==1).sum())==5
        and int((links>0).sum())==6,
        "sealed nine human TRAIN graphs counts drift")
    return TrainOnlyTargets(ev,role,polar,coref,links)


def join_genuine_original43m_train9(feature_file: Path, report_file: Path):
    tensors,meta,physical_sha=load_real_new9_feature_tensors(
        feature_file,report_file)
    texts=s19_train_texts()
    inputs=assemble_gold_blind_338(texts,tensors["token_hidden320"],meta)
    gold=make_train9_targets(inputs)
    manifest=train_only_manifest()
    require(manifest["new_train_annotations_sha256"]==TRAIN_ONLY_FROZEN_SHA256,
            "sealed human source must be identical")
    return JoinedGraphTrain9(inputs,gold,TRAIN_ONLY_FROZEN_SHA256,
                             physical_sha,True)


def train9_source_manifest(j: JoinedGraphTrain9):
    require(j.graph_sha==TRAIN_ONLY_FROZEN_SHA256
            and j.original_j7_parent_616_proven,
            "new9 original43M TRAIN source SHA drift")
    return {
        "stage":"NRA3_S19F_TRUE_ORIGINAL43M_GT_TRAIN9_TYPED_GRAPH_TARGETS_ONLY",
        "original_new9_cpu_run": "S19E_ACTUAL_CI_REQUIRED",
        "TRAIN9_source_gold_SHA256":j.graph_sha,
        "original43M_GT_new9_real_hidden_SHA256":j.physical_hidden_sha,
        "train_only_sentences":9,
        "all_real_Calibri_word_vectors":int(j.inputs.valid_word_mask.sum()),
        "supervised_events":int((j.targets.event_status>0).sum()),
        "positive_and_rejected_pred_argument_targets":
             int((j.targets.predicate_argument_role>0).sum()),
        "REJECTED_argument_links":int((j.targets.argument_polarity==1).sum()),
        "supervised_coreference_edges":int((j.targets.coreference==1).sum()),
        "supervised_event_relation_edges":int((j.targets.event_links>0).sum()),
        "all_model_features_from_true43M_GT_hidden320_and_raw_text_only":True,
        "all_human_gold_event_role_scope_coref_relations_target_only":True,
        "no_optimizer_no_new_model_scoring":True,
        "no_spent_independent_EVAL32_EVAL128_reused":True,
        "parent_616_J7_generation_digest_unchanged":True,
        "candidate_active_J7_CPU_bpp_NOT_CERTIFIED":True,
        "original_LIVE_Render_Supabase_Jessica_APK_unchanged":True,
        "output_gate":0,
        "candidate_accepted":False,
        "activation_allowed":False,
    }
