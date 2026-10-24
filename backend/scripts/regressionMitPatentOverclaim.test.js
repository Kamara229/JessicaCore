import test from "node:test";
import assert from "node:assert/strict";
import {checkMitPatentOverclaim} from "../core/validator/mitPatentOverclaim.js";
import {validateClaimEvidence} from "../core/validator/claimEvidenceValidator.js";

const question="Сравни патентные условия MPL 2.0 и MIT License по официальным текстам.";
const bad=[
  "MIT License",
  "В официальном тексте MIT License нет явно сформулированной патентной лицензии.",
  "Текст лицензии состоит из одного абзаца.",
  "Патентные права не охраняются в рамках лицензии."
].join("\n");
const safe=[
  "MIT License: прямой патентный грант в тексте не указан.",
  "Отсутствие прямой оговорки не доказывает отсутствие подразумеваемых патентных прав.",
  "MIT не содержит нумерованных пунктов: предоставление прав, условие об уведомлениях и отказ от гарантий."
].join("\n");

test("refuse ungrounded MIT patent-rights absence and one-paragraph characterization",()=>{
 const result=checkMitPatentOverclaim(question,bad);
 assert.equal(result?.valid,false);
 assert.equal(result?.shouldRetry,true);
 assert.match(result.reason,/отсутствие|одного абзаца/i);
});
test("MIT absence of express grant is not falsely rejected",()=>{
 assert.equal(checkMitPatentOverclaim(question,safe),null);
 assert.equal(checkMitPatentOverclaim("Сравни Apache и MPL патенты",bad),null);
});
test("reject unsupported MIT language before contacting claim validator",async()=>{
 const result=await validateClaimEvidence(question,{evidence:{mode:"source_content"}},
  {results:[]},{text:bad});
 assert.equal(result.valid,false);
 assert.equal(result.shouldRetry,true);
 assert.match(result.reason,/патентных прав|MIT License/i);
});

test("valid explanation is not rejected for saying one cannot conclude rights do not exist",()=>{
 const long=[
 "Нет. Отсутствие прямо сформулированного патентного гранта в MIT License не доказывает отсутствие патентных прав.",
 "В тексте перечислены права использовать, копировать, изменять и распространять программу.",
 "Указаны сохранение уведомлений и отказ от гарантий.",
 "Из молчания текста нельзя заключить, что патентных прав не существует либо что подразумеваемая лицензия невозможна.",
 "Этот вопрос требует отдельного анализа применимого патентного права."
 ].join(" ");
 assert.equal(checkMitPatentOverclaim(question,long),null);
 const wrong="MIT License не содержит отдельного патентного гранта, поэтому патентные права не охраняются в рамках лицензии. "+
   "Но вопрос не равен существованию патентных прав. Остальные разрешения перечислены в её тексте. "+
   "Для такого вывода недостаточно самого текста. Документ содержит сохранение уведомлений и отказ от гарантий.";
 assert.equal(checkMitPatentOverclaim(question,wrong)?.valid,false);
});
