import test from "node:test";
import assert from "node:assert/strict";
import {buildPatternEvidence} from "../core/learning/patternExtraction/patternEvidence.js";
import {hasVerifiedPatternEvidence} from "../core/learning/patternExtraction/patternEvidencePolicy.js";

const task="Сравни лицензии и подтверди выводы официальными источниками";
const first="https://www.apache.org/licenses/LICENSE-2.0.txt";
const second="https://opensource.org/license/bsd-3-clause";
const table="Сравнение по каждому критерию. ".repeat(110);
function trace(answer){return {task,result:{success:true,verified:true,status:"COMPLETED",
 validation:{valid:true,degraded:false,outcomeType:"result"},answer:{text:answer}}};}
test("preserve both original citations after a long comparison",()=>{
 const answer=table+"\n\nИсточники:\n"+first+"\n"+second;
 assert.ok(answer.length>3000);
 const evidence=buildPatternEvidence({trace:trace(answer)});
 assert.ok(evidence.answer.includes(first));
 assert.ok(evidence.answer.includes(second));
 assert.equal(hasVerifiedPatternEvidence(evidence),true);
});
test("do not treat an uncited comparison as a learning success",()=>{
 assert.equal(hasVerifiedPatternEvidence(buildPatternEvidence({trace:trace(table)})),false);
});
