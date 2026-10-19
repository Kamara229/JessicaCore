import test from "node:test";
import assert from "node:assert/strict";
import {guardComparisonUnknowns} from "../core/composer/comparisonUnknownGuard.js";
import {containsExplicitNoticeDuty} from "../core/validator/claim/comparisonExplicitDuties.js";

const urls=["https://www.apache.org/licenses/LICENSE-2.0.html",
 "https://opensource.org/license/bsd-3-clause"];
const task="Сравни две лицензии по сохранению уведомлений, используя официальные тексты.";
const table=[
 "| Критерий | Apache License 2.0 | BSD 3-Clause |",
 "| --- | --- | --- |",
 "| Уведомления и авторство | Требуется сохранять авторские уведомления | Требуется сохранять авторские уведомления |"
].join("\n");
function sources(textA,textB) {
 return {results:[
  {tool:"web_fetch",success:true,data:{url:urls[0],
   title:"Apache License 2.0",content:textA}},
  {tool:"web_fetch",success:true,data:{url:urls[1],
   title:"BSD 3-Clause",content:textB}}
 ]};
}
const apache="TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION "+
 "You must retain, in the Source form of any Derivative Works that You distribute, "+
 "all copyright, patent, trademark, and attribution notices. "+
 "END OF TERMS AND CONDITIONS";
const bsd="Redistributions of source code must retain the above copyright notice, "+
 "this list of conditions and the following disclaimer.";

test("actual source obligations are never replaced by unverifiable placeholders",()=>{
 assert.equal(containsExplicitNoticeDuty(apache),true);
 assert.equal(containsExplicitNoticeDuty(bsd),true);
 const guarded=guardComparisonUnknowns(task,{evidence:{mode:"source_content"}},
   sources(apache,bsd),table);
 assert.equal(guarded.text,table);
 assert.equal(guarded.changes,0);
});
test("no relevant source duty must not become invented proof",()=>{
 assert.equal(containsExplicitNoticeDuty("You may add your own copyright statement"),false);
});
