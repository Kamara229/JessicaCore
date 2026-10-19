import test from "node:test";
import assert from "node:assert/strict";
import {containsExplicitNoticeDuty} from "../core/validator/claim/comparisonExplicitDuties.js";
import {findAvoidableUnknownComparisonCell} from "../core/validator/claim/comparisonAvoidableUnknown.js";
import {attributeComparisonSources} from "../core/composer/comparisonSourceAttribution.js";

const apache="https://www.apache.org/licenses/LICENSE-2.0.html";
const bsd="https://opensource.org/license/bsd-3-clause";
const textA="TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION "+
 "You must retain, in the Source form of any Derivative Works that You distribute, "+
 "all copyright, patent, trademark, and attribution notices from the Source form of the Work. "+
 "END OF TERMS AND CONDITIONS";
const textB="Redistributions of source code must retain the above copyright notice, "+
 "this list of conditions and the following disclaimer. "+
 "Redistributions in binary form must reproduce the above copyright notice.";
const run={results:[
 {tool:"web_fetch",success:true,data:{url:apache,content:textA}},
 {tool:"web_fetch",success:true,data:{url:bsd,content:textB}}
]};
const answer=[
 "| Критерий | Apache License 2.0 | BSD 3-Clause |",
 "| --- | --- | --- |",
 "| Уведомления и авторство | Не удалось подтвердить по прочитанному тексту | Сохранять copyright |",
 "| Патентные условия | Есть патентная лицензия | Не удалось подтвердить по прочитанному тексту |"
].join("\n");

test("explicit notice retention is recognized from an operative source",()=>{
 assert.equal(containsExplicitNoticeDuty(textA),true);
 assert.equal(containsExplicitNoticeDuty(textB),true);
 assert.equal(containsExplicitNoticeDuty("You may add Your own copyright statement"),false);
});

test("a genuinely confirmable duty cannot remain unknown",()=>{
 const result=findAvoidableUnknownComparisonCell("Сравни уведомления и патентные условия",answer,run.results);
 assert.ok(result);
 assert.match(result.reason,/не удалось подтвердить/);
 assert.equal(result.sourceUrl,apache);
});

test("honest uncertainty about unproven patents remains allowed",()=>{
 const corrected=answer.replace(
  "Не удалось подтвердить по прочитанному тексту | Сохранять copyright",
  "Сохранять copyright | Сохранять copyright"
 );
 assert.equal(findAvoidableUnknownComparisonCell(
   "Сравни уведомления и патентные условия",corrected,run.results),null);
});

test("official source links replace internal references",()=>{
 const original="Разрешено распространять【S1-use-1】; сохранять уведомления【S2-notices-1】.";
 const output=attributeComparisonSources(original,run);
 assert.equal(output.text.includes("【S1-"),false);
 assert.ok(output.text.includes(apache));
 assert.ok(output.text.includes(bsd));
 assert.ok(output.text.includes("[Официальный источник 1]"));
});

test("source links are added even when model forgets citations",()=>{
 const first=attributeComparisonSources(answer,run);
 assert.ok(first.text.includes(apache)&&first.text.includes(bsd));
 const second=attributeComparisonSources(first.text,run);
 assert.equal(second.text,first.text);
});

test("never invent links when fetched sources are incomplete",()=>{
 assert.deepEqual(attributeComparisonSources("Text",{results:[]}),{text:"Text",changes:0});
});
