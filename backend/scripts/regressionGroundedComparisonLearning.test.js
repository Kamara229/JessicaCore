import test from "node:test";
import assert from "node:assert/strict";
import {validateGroundedComparisonLearning} from "../core/learning/quality/groundedComparisonLearningGate.js";

const task="Сравни две лицензии. Найди официальные тексты и подтверди выводы официальными источниками.";
function proposal(text,{verified=true,status="COMPLETED"}={}) {
 return {proposedExperience:{examples:[{
  task,success:true,result:{success:true,verified,status,text}
 }]}};
}
test("legacy completed comparisons without official citations cannot become a Skill",()=>{
 const result=validateGroundedComparisonLearning(proposal(
   "| Критерий | A | B |\n|---|---|---|\n| Уведомления | Не удалось подтвердить | Сохранить |"));
 assert.equal(result.passed,false);
});
test("two distinct cited official documents permit ordinary autonomous quality review",()=>{
 const result=validateGroundedComparisonLearning(proposal(
   "| Критерий | A | B |\n|---|---|---|\n| Уведомления | Сохранить [1](https://example.org/lic-a) | Сохранить [2](https://example.org/lic-b) |"));
 assert.equal(result.passed,true);
});
test("duplicating one reference is not two sources",()=>{
 assert.equal(validateGroundedComparisonLearning(proposal(
  "https://example.org/lic-a https://example.org/lic-a")).passed,false);
});
test("unverified comparison evidence is never accepted as a learning example",()=>{
 assert.equal(validateGroundedComparisonLearning(proposal(
  "https://example.org/lic-a https://example.org/lic-b",
  {verified:false})).passed,false);
});
test("unrelated Skills continue through their normal quality inspector",()=>{
 assert.equal(validateGroundedComparisonLearning({proposedExperience:{examples:[
  {task:"Calculate a date interval",success:true,result:{text:"9 days"}}
 ]}}).passed,true);
});
