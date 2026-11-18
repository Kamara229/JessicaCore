"""S38 NEW 32-sentence Russian adversarial grammar challenge, frozen at commit.

Challenge is POST-HOC to S35/S37 policy development, but these exact
32 texts have not appeared in previous 256 synthetic TRAIN graphs. It
is *not* a statistically independent blind model evaluation: classes
are adversarially designed with knowledge of S37's raw18 gate.
Labels are only for OFFLINE rule diagnostic after input-only inference.

Groups:
  A 8 local НЕ + dative recipients, regular finite predicates
  B 8 nonlocal negation recipients, valid excluded recipients
  C 8 passive/infinitive source predicates
  D 8 quoted/mentioned phrases; НЕ+dative is NOT event recipient

No source texts, gold, labels or IDs enter Calibri/NRA forward. S38
first tests rule-only semantic coverage; a separate later *new true
43M native CPU* capture and model-head scoring would be required to
claim S37 neural generalization on these exact texts.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar

STAGE="NRA3_S38_NEW_POSTHOC_RUSSIAN_NEGATION_CHALLENGE_SOURCE32_V1"

@dataclass(frozen=True)
class ChallengeRow:
    key: str
    text: str
    predicate: str
    recipient: str
    rejected: bool
    phenomenon: str

# Predeclared human-curated labels. Do NOT select / tune policy based on
# any held TRAIN or fresh S38 neural predictions. One source, one target
# per example; readings are by explicit semantic intent.
ROWS=(
  ChallengeRow("A01","Марта передала акт не соседу.","передала","соседу",True,"local_finite"),
  ChallengeRow("A02","Павел отправит чек не преподавателю.","отправит","преподавателю",True,"local_finite"),
  ChallengeRow("A03","Рита показала фото не знакомому.","показала","знакомому",True,"local_finite"),
  ChallengeRow("A04","Дима доставил посылку не редактору.","доставил","редактору",True,"local_finite"),
  ChallengeRow("A05","Нина вручила ключ не мастеру.","вручила","мастеру",True,"local_finite"),
  ChallengeRow("A06","Ольга направит справку не адвокату.","направит","адвокату",True,"local_finite"),
  ChallengeRow("A07","Артём отдал список не менеджеру.","отдал","менеджеру",True,"local_finite"),
  ChallengeRow("A08","Лера отправила письмо не бухгалтеру.","отправила","бухгалтеру",True,"local_finite"),

  ChallengeRow("B01","Никому из коллег Марта не передала акт.","передала","Никому",True,"nonlocal_negative"),
  ChallengeRow("B02","Никому Павел не отправит чек.","отправит","Никому",True,"nonlocal_negative"),
  ChallengeRow("B03","Марта передала акт вовсе не тому соседу.","передала","соседу",True,"nonlocal_negative"),
  ChallengeRow("B04","Рита не передала фото ни одному соседу.","передала","соседу",True,"nonlocal_negative"),
  ChallengeRow("B05","Не каждому курьеру Павел отправил квитанцию.","отправил","курьеру",True,"nonlocal_negative"),
  ChallengeRow("B06","Ни одному диспетчеру Нина не показала документы.","показала","диспетчеру",True,"nonlocal_negative"),
  ChallengeRow("B07","Игорь передал сведения совершенно не тому врачу.","передал","врачу",True,"nonlocal_negative"),
  ChallengeRow("B08","Лена передала запись не самому опытному мастеру.","передала","мастеру",True,"nonlocal_negative"),

  ChallengeRow("C01","Письмо доставлено не получателю.","доставлено","получателю",True,"nonfinite_predicate"),
  ChallengeRow("C02","Квитанция выслана не кассиру.","выслана","кассиру",True,"nonfinite_predicate"),
  ChallengeRow("C03","Задание вручено не ученику.","вручено","ученику",True,"nonfinite_predicate"),
  ChallengeRow("C04","Запись передана не врачу.","передана","врачу",True,"nonfinite_predicate"),
  ChallengeRow("C05","Мария решила передать акт не коллеге.","передать","коллеге",True,"nonfinite_predicate"),
  ChallengeRow("C06","Владимир успел вручить письмо не соседу.","вручить","соседу",True,"nonfinite_predicate"),
  ChallengeRow("C07","Лена может показать запись не оператору.","показать","оператору",True,"nonfinite_predicate"),
  ChallengeRow("C08","Кирилл должен отправлять отчёт не директору.","отправлять","директору",True,"nonfinite_predicate"),

  ChallengeRow("D01","Марта произнесла выражение «не соседу».","произнесла","соседу",False,"quoted_negative_mention"),
  ChallengeRow("D02","Павел напечатал фразу «не редактору».","напечатал","редактору",False,"quoted_negative_mention"),
  ChallengeRow("D03","Рита прочитала цитату «не курьеру».","прочитала","курьеру",False,"quoted_negative_mention"),
  ChallengeRow("D04","Дима исправил запись «не директору».","исправил","директору",False,"quoted_negative_mention"),
  ChallengeRow("D05","Нина записала сочетание «не бухгалтеру».","записала","бухгалтеру",False,"quoted_negative_mention"),
  ChallengeRow("D06","Ольга переписала текст «не оператору».","переписала","оператору",False,"quoted_negative_mention"),
  ChallengeRow("D07","Лера выделила слова «не соседу».","выделила","соседу",False,"quoted_negative_mention"),
  ChallengeRow("D08","Артём процитировал строку «не врачу».","процитировал","врачу",False,"quoted_negative_mention"),
)

def canonical_data_bytes():
    return json.dumps([
        {"key":x.key,"text":x.text,"source":x.predicate,
         "target":x.recipient,"semantic_gold_rejected":x.rejected,
         "phenomenon":x.phenomenon}
        for x in ROWS
    ],sort_keys=True,ensure_ascii=False,separators=(",",":")).encode("utf-8")

def challenge_digest():
    return hashlib.sha256(canonical_data_bytes()).hexdigest()

def validate_challenge():
    assert len(ROWS)==32 and len({r.text for r in ROWS})==32
    assert len({r.key for r in ROWS})==32
    classes={q:sum(r.phenomenon==q for r in ROWS) for q in (
        "local_finite","nonlocal_negative","nonfinite_predicate","quoted_negative_mention")}
    assert classes=={q:8 for q in classes}
    for r in ROWS:
        toks=raw_surface_grammar(r.text)
        assert sum(x.form==r.predicate for x in toks)==1,(r.key,"source ambiguous")
        assert sum(x.form==r.recipient for x in toks)==1,(r.key,"target ambiguous")
        assert r.predicate!=r.recipient
        assert r.text==r.text.strip()
        assert r.rejected==(r.phenomenon!="quoted_negative_mention")
    return {"stage":STAGE,"count":len(ROWS),"group_counts":classes,
            "positive":sum(x.rejected for x in ROWS),
            "negative":sum(not x.rejected for x in ROWS),
            "canonical_source_and_label_sha256":challenge_digest()}

if __name__=="__main__":
    print(json.dumps(validate_challenge(),ensure_ascii=False,sort_keys=True))
