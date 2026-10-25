import test from "node:test";
import assert from "node:assert/strict";
import "../tools/webFetchTool.js";
import {groundedApachePatentSection} from "../core/composer/groundedApachePatentSection.js";
import {composeAnswer} from "../core/answerComposer.js";
import {validateResult} from "../core/validator.js";
import {tryNativeDecomposition} from "../core/decomposer/nativeDecomposer.js";
import {composeComplexAnswer} from "../core/complexAnswerComposer.js";

const userTask="Сколько дней с 27 февраля по 2 марта 2024 года (27-е исключить, 2-е включить)? Затем укажи пункт о патентной лицензии Apache 2.0 и официальный URL.";
const url="https://www.apache.org/licenses/LICENSE-2.0.txt";
const sample=[
 "Apache License Version 2.0, January 2004.",
 "3. Grant of Patent License. Subject to the terms and conditions of this License,",
 "each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge,",
 "royalty-free, irrevocable (except as stated in this section) patent license to make, have made, use,",
 "offer to sell, sell, import, and otherwise transfer the Work.",
 "If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit)",
 "alleging that the Work or a Contribution incorporated within the Work constitutes direct or contributory patent infringement,",
 "then any patent licenses granted to You under this License for that Work shall terminate as of the date such litigation is filed.",
 "4. Redistribution. You may reproduce and distribute copies."
].join(" ");
const plan={intent:"native_single_license_primary_text",metadata:{licenseId:"Apache-2.0"},
 evidence:{mode:"source_content"},steps:[{id:"fetch_named_license",tool:"web_fetch",arguments:{url}}]};
const run=(content=sample)=>({success:true,results:[{tool:"web_fetch",success:true,
 data:{url,content,truncated:false}}]});

test("explicit sequence retains date and Apache section task separately",()=>{
 const d=tryNativeDecomposition({task:userTask});
 assert.equal(d.handled,true);
 assert.equal(d.decomposition.subtasks.length,2);
 assert.match(d.decomposition.subtasks[0].text,/27 февраля/);
 assert.match(d.decomposition.subtasks[1].text,/патентной лицензии Apache 2\.0/);
});

test("official section is answered and validated without calling the AI model",async()=>{
 const task="укажи пункт о патентной лицензии Apache 2.0 и официальный URL.";
 const direct=groundedApachePatentSection(task,plan,run());
 assert.equal(direct?.source,"bounded-apache-patent-section");
 assert.match(direct.text,/разделе 3/u);
 assert.match(direct.text,/Grant of Patent License/u);
 assert.match(direct.text,/https:\/\/www\.apache\.org\/licenses\/LICENSE-2\.0\.txt/u);
 assert.deepEqual(await composeAnswer(task,plan,run()),direct);
 const checked=await validateResult(task,plan,run(),direct);
 assert.equal(checked.valid,true,checked.reason);
 assert.equal(checked.validationMode,"bounded-primary-text");
});

test("official primary-source section refuses missing or altered patent grant",()=>{
 const task="укажи пункт о патентной лицензии Apache 2.0 и официальный URL.";
 assert.equal(groundedApachePatentSection(task,plan,run(sample.replace("Each Contributor hereby grants","Each Contributor perhaps grants"))),null);
 assert.equal(groundedApachePatentSection(task,plan,run(sample.replace("3. Grant of Patent License.","3. Unknown Clause."))),null);
 assert.equal(groundedApachePatentSection(task,plan,{results:[]}),null);
});

test("date and section responses combine without inventing missing information",async()=>{
 const joined=await composeComplexAnswer(userTask,{},{
 results:[
  {id:1,status:"COMPLETED",success:true,verified:true,result:"От 2024-02-27 до 2024-03-02 прошло 4 полных календарных дня."},
  {id:2,status:"COMPLETED",success:true,verified:true,result:groundedApachePatentSection(
    "укажи пункт о патентной лицензии Apache 2.0 и официальный URL.",plan,run()).text}
 ]
 });
 assert.equal(joined.source,"deterministic-results");
 assert.match(joined.text,/4 полных календарных дня/);
 assert.match(joined.text,/разделе 3/);
});
