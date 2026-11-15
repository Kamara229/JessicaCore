"""NRA2 3.0 S4: independent hand-annotated Russian UD *TRAIN ONLY* source.

Development TRAIN examples are deliberately unrelated to earlier English
R0–R5 graph EVALs. They are not an independent holdout: they are permitted
to supervise grammar and semantic roles only after real Calibri hidden320
has been captured and SHA-pinned in a later, separately reviewed stage.

All three levels remain inside the unified NRA3 model. Grammatical subject
is never assumed equivalent to semantic source/agent: 'получил от' and
passives explicitly differ. Pronouns have roles but NO coreference gold.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from typing import Literal

from research.nra2_bridge.stage3ck_russian_ud_role_bridge import (
    UDWord,extract_grammar,GrammarContractError,
)
from research.nra3_s0.neural_area import MORPH, SEMANTIC_ROLES, UPOS

TRAIN_SPLIT="NRA3_S4_RU_UD_TRAIN_ONLY"
# Physically obtained from S4 CPU CI #43698097398; immutable TRAIN-only
# annotation+offset canonical digest. Any edits require a NEW research stage.
TRAIN12_CANONICAL_SHA256="6ee1f270d413f4cfc79727bd48386ee29881520fcf6834309869733fd00f7a61"
# (surface, UPOS, head, deprel, morphological-features, semantic role)
Spec=tuple[str,str,int,str,tuple[str,...],str]

# TRAIN annotation only: explicit dictionary lemmas, not a runtime
# morphological oracle. At inference the neural area must predict structure.
_TRAIN_LEMMA={
    "анна":"Анна","отправила":"отправить","павлу":"Павел",
    "письмо":"письмо",".":".","павел":"Павел",
    "получил":"получить","от":"от","анны":"Анна",
    "борис":"Борис","передал":"передать","лене":"Лена",
    "папку":"папка","лена":"Лена","получила":"получить",
    "бориса":"Борис","курьер":"курьер","доставил":"доставить",
    "ирине":"Ирина","посылку":"посылка","ирина":"Ирина",
    "курьера":"курьер","внимательный":"внимательный",
    "было":"быть","отправлено":"отправить","анной":"Анна",
    "папка":"папка","была":"быть","передана":"передать",
    "борисом":"Борис",
}


@dataclass(frozen=True)
class RussianTrainSentence:
    text:str
    words:tuple[UDWord,...]
    roles:tuple[str,...]
    split:str=TRAIN_SPLIT

    def sha256(self)->str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def _row(text:str, specifications:tuple[Spec,...])->RussianTrainSentence:
    if not isinstance(text,str) or not text.endswith("."):
        raise ValueError("S4 requires Russian period-terminated source")
    words=[]
    roles=[]
    end=0
    for i,(surface,upos,head,dep,features,role) in enumerate(specifications,1):
        start=text.find(surface,end)
        if start<end or start<0:
            raise ValueError("S4 surface must exist sequentially in exact text")
        after=start+len(surface)
        try:
            lemma=_TRAIN_LEMMA[surface.casefold()]
        except KeyError as exc:
            raise ValueError("S4 TRAIN-only lemma must be explicitly annotated") from exc
        words.append(UDWord(i,surface,lemma,upos,head,dep,
                            start,after,features))
        roles.append(role)
        end=after
    row=RussianTrainSentence(text,tuple(words),tuple(roles))
    frame=extract_grammar(row.text,row.words)
    if (frame.semantic_direction is not None or frame.graph_reachability is not None
            or row.split!=TRAIN_SPLIT
            or len(row.roles)!=len(row.words)
            or not set(row.roles).issubset(set(SEMANTIC_ROLES))
            or not any(x=="AGENT" or x=="SOURCE" for x in row.roles)
            or not any(x=="PATIENT" for x in row.roles)
            or not any(x=="RECIPIENT" for x in row.roles)
            or not any(w.upos=="VERB" for w in row.words)
            or any(f not in MORPH for w in row.words for f in w.feats)):
        raise ValueError("S4 grammar/semantics TRAIN source contract rejected")
    return row


def build_train()->tuple[RussianTrainSentence,...]:
    """Small explicitly TRAIN-only examples, never use as independent EVAL."""
    N=("Case=Nom",)
    D=("Case=Dat",)
    A=("Case=Acc",)
    G=("Case=Gen",)
    P=("Voice=Pass",)
    # Case features are only those used by this deliberately narrow pilot.
    # Rule-based labels are NOT fed as inference inputs; future text must
    # be parsed by the model itself after training.
    rows=(
      _row("Анна отправила Павлу письмо.",(
         ("Анна","PROPN",2,"nsubj",N,"AGENT"),
         ("отправила","VERB",0,"root",(),"NONE"),
         ("Павлу","PROPN",2,"iobj",D,"RECIPIENT"),
         ("письмо","NOUN",2,"obj",A,"PATIENT"),
         (".","PUNCT",2,"punct",(),"NONE"))),
      _row("Письмо Павлу отправила Анна.",(
         ("Письмо","NOUN",3,"obj",A,"PATIENT"),
         ("Павлу","PROPN",3,"iobj",D,"RECIPIENT"),
         ("отправила","VERB",0,"root",(),"NONE"),
         ("Анна","PROPN",3,"nsubj",N,"AGENT"),
         (".","PUNCT",3,"punct",(),"NONE"))),
      _row("Павел получил письмо от Анны.",(
         ("Павел","PROPN",2,"nsubj",N,"RECIPIENT"),
         ("получил","VERB",0,"root",(),"NONE"),
         ("письмо","NOUN",2,"obj",A,"PATIENT"),
         ("от","ADP",5,"case",(),"NONE"),
         ("Анны","PROPN",2,"obl",G,"SOURCE"),
         (".","PUNCT",2,"punct",(),"NONE"))),
      _row("Борис передал Лене папку.",(
         ("Борис","PROPN",2,"nsubj",N,"AGENT"),
         ("передал","VERB",0,"root",(),"NONE"),
         ("Лене","PROPN",2,"iobj",D,"RECIPIENT"),
         ("папку","NOUN",2,"obj",A,"PATIENT"),
         (".","PUNCT",2,"punct",(),"NONE"))),
      _row("Папку Лене передал Борис.",(
         ("Папку","NOUN",3,"obj",A,"PATIENT"),
         ("Лене","PROPN",3,"iobj",D,"RECIPIENT"),
         ("передал","VERB",0,"root",(),"NONE"),
         ("Борис","PROPN",3,"nsubj",N,"AGENT"),
         (".","PUNCT",3,"punct",(),"NONE"))),
      _row("Лена получила папку от Бориса.",(
         ("Лена","PROPN",2,"nsubj",N,"RECIPIENT"),
         ("получила","VERB",0,"root",(),"NONE"),
         ("папку","NOUN",2,"obj",A,"PATIENT"),
         ("от","ADP",5,"case",(),"NONE"),
         ("Бориса","PROPN",2,"obl",G,"SOURCE"),
         (".","PUNCT",2,"punct",(),"NONE"))),
      _row("Курьер доставил Ирине посылку.",(
         ("Курьер","NOUN",2,"nsubj",N,"AGENT"),
         ("доставил","VERB",0,"root",(),"NONE"),
         ("Ирине","PROPN",2,"iobj",D,"RECIPIENT"),
         ("посылку","NOUN",2,"obj",A,"PATIENT"),
         (".","PUNCT",2,"punct",(),"NONE"))),
      _row("Посылку Ирине доставил курьер.",(
         ("Посылку","NOUN",3,"obj",A,"PATIENT"),
         ("Ирине","PROPN",3,"iobj",D,"RECIPIENT"),
         ("доставил","VERB",0,"root",(),"NONE"),
         ("курьер","NOUN",3,"nsubj",N,"AGENT"),
         (".","PUNCT",3,"punct",(),"NONE"))),
      _row("Ирина получила посылку от курьера.",(
         ("Ирина","PROPN",2,"nsubj",N,"RECIPIENT"),
         ("получила","VERB",0,"root",(),"NONE"),
         ("посылку","NOUN",2,"obj",A,"PATIENT"),
         ("от","ADP",5,"case",(),"NONE"),
         ("курьера","NOUN",2,"obl",G,"SOURCE"),
         (".","PUNCT",2,"punct",(),"NONE"))),
      _row("Внимательный курьер доставил Ирине посылку.",(
         ("Внимательный","ADJ",2,"amod",N,"NONE"),
         ("курьер","NOUN",3,"nsubj",N,"AGENT"),
         ("доставил","VERB",0,"root",(),"NONE"),
         ("Ирине","PROPN",3,"iobj",D,"RECIPIENT"),
         ("посылку","NOUN",3,"obj",A,"PATIENT"),
         (".","PUNCT",3,"punct",(),"NONE"))),
      _row("Письмо было отправлено Анной Павлу.",(
         ("Письмо","NOUN",3,"nsubj:pass",N,"PATIENT"),
         ("было","AUX",3,"aux",(),"NONE"),
         ("отправлено","VERB",0,"root",P,"NONE"),
         ("Анной","PROPN",3,"obl:agent",("Case=Ins",),"AGENT"),
         ("Павлу","PROPN",3,"iobj",D,"RECIPIENT"),
         (".","PUNCT",3,"punct",(),"NONE"))),
      _row("Папка была передана Борисом Лене.",(
         ("Папка","NOUN",3,"nsubj:pass",N,"PATIENT"),
         ("была","AUX",3,"aux",(),"NONE"),
         ("передана","VERB",0,"root",P,"NONE"),
         ("Борисом","PROPN",3,"obl:agent",("Case=Ins",),"AGENT"),
         ("Лене","PROPN",3,"iobj",D,"RECIPIENT"),
         (".","PUNCT",3,"punct",(),"NONE"))),
    )
    validate_train(rows)
    return rows


def validate_train(rows:tuple[RussianTrainSentence,...])->None:
    if (len(rows)!=12 or any(r.split!=TRAIN_SPLIT for r in rows)
            or len({r.sha256() for r in rows})!=len(rows)):
        raise ValueError("NRA3 S4 split drift or duplicate Russian TRAIN text")
    for row in rows:
        if not isinstance(row,RussianTrainSentence):
            raise ValueError("NRA3 S4 invalid TRAIN row")
        frame=extract_grammar(row.text,row.words)
        if frame.semantic_direction is not None:
            raise ValueError("NRA3 S4 forbidden direction oracle")
        # No semantic-role inference from grammatical subject. Receive and
        # passive are explicitly counterexamples to subject=AGENT.
        if "получил" in row.text or "получила" in row.text:
            if row.roles[next(i for i,w in enumerate(row.words)
                              if w.deprel=="nsubj")]!="RECIPIENT":
                raise ValueError("NRA3 S4 recipient subject not protected")
        for w,role in zip(row.words,row.roles):
            if w.upos=="PUNCT" and role!="NONE":
                raise ValueError("NRA3 S4 punctuation may not carry semantics")
    if sum(any(w.deprel=="nsubj:pass" for w in r.words) for r in rows)!=2:
        raise ValueError("NRA3 S4 passive examples lost")


def commitment(rows:tuple[RussianTrainSentence,...])->dict:
    validate_train(rows)
    canonical=[{
      "split":r.split,"text":r.text,"text_sha256":r.sha256(),
      "tokens":[{"id":w.word_id,"form":w.text,"lemma":w.lemma,"upos":w.upos,
                 "head":w.head,"deprel":w.deprel,
                 "start":w.start,"end":w.end,"feats":w.feats,
                 "semantic_role":r.roles[i]}
                for i,w in enumerate(r.words)],
      "no_coreference_labels":True,"no_graph_truth":True,
    } for r in rows]
    serialized=json.dumps(canonical,sort_keys=True,ensure_ascii=False,
                          separators=(",",":")).encode("utf-8")
    digest=hashlib.sha256(serialized).hexdigest()
    # Immutable complete source was verified by independent S4 CPU CI
    # #43698097398. Any future alteration needs a NEW research stage.
    if digest!=TRAIN12_CANONICAL_SHA256:
        raise ValueError("S4 immutable TRAIN12 annotation SHA changed")
    return {
      "stage":"NRA3_S4_HAND_ANNOTATED_RUSSIAN_UD_TRAIN_ONLY",
      "split":TRAIN_SPLIT,
      "row_count":len(rows),
      "canonical_training_rows_sha256":digest,
      "ordered_prompt_sha256":[r.sha256() for r in rows],
      "independent_EVAL_count":0,
      "spent_five_EVAL128_not_loaded":True,
      "reserved_sixth_S3CI_EVAL128_not_loaded":True,
      "actual_43M_Calibri_hidden320_captured":False,
      "QAT_training_performed":False,
      "native_full_EOS_616_J7_verified":False,
      "output_gate":0,
      "candidate_accepted":False,
    }
