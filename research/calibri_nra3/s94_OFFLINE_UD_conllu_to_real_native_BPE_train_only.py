"""NRA3 S94: constrained OFFLINE UD (CoNLL-U) -> native Calibri BPE TRAIN targets.

This is a TRAIN LABEL ADAPTER, NOT a runtime parser or sidecar inference
service. No external Stanza/UDPipe called by Calibri. A pre-annotated
Russian UD sentence, not model output, is used to construct supervised
token-role arcs. Training teacher noise is NOT human-verified truth.

CRITICAL: source-only real Calibri original BPE IDs, Unicode offsets
and raw word spans must agree exactly. Each grammatical word maps to
its FINAL original native BPE piece, NEVER to guessed split positions.
Reject shared BPE / template-crossing or ambiguous source anchors.
Unknown annotations stay IGNORE/UNKNOWN, never fabricated negatives.

Only *unambiguous* UD relations are projected: nsubj->actor;
nsubj:pass->patient; obj->patient; iobj->recipient;
obl:agent->actor; source negator particle attached to verb as
advmod:neg/advmod -> negation_scope. Passive Voice=Pass or nsubj:pass
annotates passive factor; attached 'не' marks local negative factor.
Other event factuality, uncertainty, reported speech, coreference,
time, cross-clause roles are NEVER manufactured from syntax alone.
New train-only source and annotation sets are required for genuine
learning; this ABI itself does NOT establish reliable Russian parsing.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from research.nra3_s19.graph_schema import raw_model_input
from research.nra3_s20.s51_fractional_native_bpe_overlap import (
    NativeOverlapAlignment,
)
from research.calibri_nra3.s91_neural_multilayer_event_graph import (
    N_ROLES,N_STATES,N_MICRO,ROLE_NAMES,STATE_NAMES,
)
from research.calibri_nra3.s93_offline_ud_grammatical_role_train_loss_ONLY import (
    NRA3TrainOnlyTargets,IGNORE,UNKNOWN,
)

STAGE="CALIBRI_NRA3_S94_OFFLINE_UD_CONLLU_NATIVE_BPE_EVENT_ROLE_TRAIN_LABELS"
ROLE_MAP={
    "nsubj":"actor",
    "nsubj:pass":"patient",
    "obj":"patient",
    "iobj":"recipient",
    "obl:agent":"actor",
}


def require(v,why):
    if not bool(v):
        raise ValueError("NRA3 S94 offline UD native original-BPE TRAIN: "+why)


@dataclass(frozen=True)
class UDWord:
    idx:int
    form:str
    upos:str
    feats:str
    head:int
    deprel:str
    begin:int
    end:int


@dataclass(frozen=True)
class UDSentence:
    text:str
    words:tuple[UDWord,...]
    evidence:str="offline_UD_teaching_annotations_NOT_independently_human_adjudicated"


def read_one_strict_conllu(conllu:str)->UDSentence:
    """A single UD sentence; fail if segmentation/source text differs."""
    require(isinstance(conllu,str) and conllu.strip()
            and "\n\n" not in conllu.strip(), "one complete CoNLL-U sentence only")
    lines=[z for z in conllu.strip().splitlines() if z.strip()]
    texts=[z.split("=",1)[1].strip() for z in lines
           if z.startswith("# text =")]
    require(len(texts)==1 and len(texts[0])>=5,
            "exact human/UD source text metadata required")
    text=texts[0];words=[];cursor=0
    for line in lines:
        if line.startswith("#"):
            continue
        cols=line.split("\t")
        require(len(cols)==10 and cols[0].isdigit()
                and cols[6].isdigit()
                and cols[1] and cols[3]
                and cols[7] and cols[7]!="_",
                "strict 10-column resolved UD integer IDs only; no MWT/empty nodes")
        idx=int(cols[0]);word=cols[1];head=int(cols[6])
        require(idx==len(words)+1 and head>=0,
                "UD words must have consecutive nonambiguous integer IDs")
        start=text.find(word,cursor)
        require(start>=0 and not text[cursor:start].strip()
                and text[start:start+len(word)]==word,
                "CoNLL-U FORM must exactly reconstruct original Unicode source")
        end=start+len(word)
        words.append(UDWord(idx,word,cols[3],cols[5],
                            head,cols[7],start,end))
        cursor=end
    require(words and not text[cursor:].strip(),
            "UD word tokens must cover full original text")
    require(all(w.head<=len(words) and w.head!=w.idx for w in words)
            and sum(w.head==0 for w in words)==1,
            "single valid UD dependency tree with exactly one root")
    return UDSentence(text,tuple(words))


def align_one_sentence_source_to_original_native_BPE(
        sentence:UDSentence,
        source_alignment:NativeOverlapAlignment)->dict[int,int]:
    """ONLY source offsets. Call BEFORE any annotated UD/gold is read.

    This helper's alignment argument is from the exact real Calibri
    original tokenizer on SOURCE, never a separately retokenized
    standalone grammar model.
    """
    raw=raw_model_input(sentence.text).tokens
    require(len(source_alignment.raw_words)==len(raw)
            and source_alignment.invalid_or_uncovered_characters==0
            and source_alignment.original_plaintext_sha256==
                __import__("hashlib").sha256(
                    sentence.text.encode("utf-8")).hexdigest(),
            "original tokenizer original source text SHA/offsets must match")
    spans={(x.start,x.end):i for i,x in enumerate(raw)}
    position={}
    used=set()
    for w in sentence.words:
        if w.upos=="PUNCT":
            continue
        i=spans.get((w.begin,w.end))
        require(i is not None and i not in used,
                "UD word segmentation diverges from S18 raw tokenizer; do NOT guess")
        used.add(i)
        slot=source_alignment.raw_words[i]
        require(slot.causally_safe
                and slot.native_token_positions
                and all(type(j) is int and j>=0 for j in slot.native_token_positions),
                "shared or template-crossing original BPE word position UNSAFE")
        position[w.idx]=slot.native_token_positions[-1]
    require(len(position)>=2
            and len(set(position.values()))==len(position),
            "two distinct words must not share a final original native BPE piece")
    return position


def offline_native_BPE_training_targets(
        sentences:tuple[UDSentence,...],
        source_alignments:tuple[NativeOverlapAlignment,...],
        source_valid:torch.Tensor)->NRA3TrainOnlyTargets:
    """Explicit UD TRAIN labels; not passed into neural model forward.

    Only roles with a VERIFIED local dependency map are annotated.
    Neutral event/factual status, other role labels and MINI-areas
    remain UNKNOWN unless a precise positive UD relation is present.
    """
    require(len(sentences)==len(source_alignments)==source_valid.shape[0]
            and source_valid.ndim==2
            and source_valid.dtype==torch.bool
            and bool(source_valid.any(-1).all()),
            "batch source-native BPE masks from real Calibri required")
    b,t=source_valid.shape
    mention=torch.full((b,t),IGNORE,dtype=torch.long)
    roles=torch.full((b,N_ROLES,t),IGNORE,dtype=torch.long)
    states=torch.full((b,t,N_STATES),float(UNKNOWN),dtype=torch.float32)
    minis=torch.full((b,t,N_MICRO),float(UNKNOWN),dtype=torch.float32)
    for batch,(sentence,align) in enumerate(zip(sentences,source_alignments)):
        valid=int(source_valid[batch].sum())
        require(valid==align.original_token_count
                and bool(source_valid[batch,:valid].all())
                and not bool(source_valid[batch,valid:].any()),
                "original native BPE valid tensor and exact source IDS length")
        native=align_one_sentence_source_to_original_native_BPE(sentence,align)
        verbs={w.idx for w in sentence.words if w.upos=="VERB"}
        for i in verbs:
            require(i in native and native[i]<valid,
                    "only fully aligned original source VERB events can be trained")
            mention[batch,native[i]]=1
        for child in sentence.words:
            h=child.head
            if h not in verbs:
                continue
            if child.idx not in native or h not in native:
                continue
            ev=native[h]
            argument=native[child.idx]
            label=ROLE_MAP.get(child.deprel)
            if (child.form.casefold()=="не" and child.upos=="PART"
                    and child.deprel in ("advmod","advmod:neg")):
                label="negation_scope"
                states[batch,ev,STATE_NAMES.index("negated")]=1.
                minis[batch,ev,1]=1.  # retained NRA2 negation mini-area
            if label is not None:
                slot=ROLE_NAMES.index(label)
                require(roles[batch,slot,ev]==IGNORE,
                        "multiple same-role argument targets require different multi-arc ABI")
                roles[batch,slot,ev]=argument
            if child.deprel=="nsubj:pass":
                states[batch,ev,STATE_NAMES.index("passive")]=1.
                minis[batch,ev,2]=1.  # NRA2 passive mini-area
        for verb in sentence.words:
            if verb.idx in verbs and "Voice=Pass" in verb.feats:
                ev=native[verb.idx]
                states[batch,ev,STATE_NAMES.index("passive")]=1.
                minis[batch,ev,2]=1.
    return NRA3TrainOnlyTargets(
        mention=mention,role_argument_token=roles,
        state=states,mini_area=minis,
        original_native_valid=source_valid.detach().clone(),
        evidence="offline_UD_teacher_only_not_independently_verified",
    )
