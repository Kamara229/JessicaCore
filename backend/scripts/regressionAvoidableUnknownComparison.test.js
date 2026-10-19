import test from "node:test";
import assert from "node:assert/strict";
import {findAvoidableUnknownComparisonCell as check} from "../core/validator/claim/comparisonAvoidableUnknown.js";

const a="https://www.apache.org/licenses/LICENSE-2.0.html";
const b="https://opensource.org/license/bsd-3-clause";
const sources=[
 {tool:"web_fetch",success:true,data:{url:a,content:"You must retain, in the Source form of any Derivative Works that You distribute, all copyright, patent, trademark, and attribution notices."}},
 {tool:"web_fetch",success:true,data:{url:b,content:"Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer."}}
];
const task="Сравни сохранение уведомлений, распространение и патентные положения";
function table(c1,c2,criterion="Уведомления и авторство",heading="| Apache 2.0 | BSD 3-Clause |"){
 return "| Критерий "+heading+"\n| --- | --- | --- |\n| "+criterion+" | "+c1+" | "+c2+" |";
}
const unknown="Не удалось подтвердить по прочитанному тексту";

test("reject unknown notices when explicit retention provision exists",()=>{
 assert.ok(check(task,table(unknown,"Сохранять уведомления"),sources));
 assert.ok(check(task,table("Сохранять уведомления",unknown),sources));
});
test("do not reject a supported comparison as avoidable unknown",()=>{
 assert.equal(check(task,table("Сохранять уведомления","Сохранять уведомления"),sources),null);
});
test("unknown patent claims stay unknown without affirmative source",()=>{
 assert.equal(check(task,table(unknown,unknown,"Патентные условия"),sources),null);
});
test("never guess a source column on unmapped headings",()=>{
 assert.equal(check(task,table(unknown,"Сохранять уведомления","Уведомления и авторство","| Первый | Второй |"),sources),null);
});
