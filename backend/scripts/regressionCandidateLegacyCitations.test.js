import test from "node:test";
import assert from "node:assert/strict";
import {hasRequestedOfficialComparisonCitations} from "../core/learning/patternExtraction/patternEvidencePolicy.js";
import {resolveLearningExampleSuccess} from "../core/learning/exampleSuccess.js";
import {mergeLearningCandidates} from "../core/learning/candidateMemory/candidateMerge.js";
const task="Сравни официальные тексты двух лицензий и подтверди выводы официальными источниками.";
const base={task,success:true,result:{success:true,verified:true,status:"COMPLETED",validation:{valid:true,degraded:false},executionMeta:{traceId:"old"}}};
const older={...base,traceId:"old",result:{...base.result,text:"Сравнение по условным ссылкам S1 и S2."}};
const newer={...base,traceId:"new",result:{...base.result,text:"Проверено по https://www.apache.org/licenses/LICENSE-2.0.txt и https://opensource.org/license/bsd-3-clause"}};
test("require direct links even when task itself did not paste URLs",()=>{
 assert.equal(hasRequestedOfficialComparisonCitations(task,older.result.text),false);
 assert.equal(hasRequestedOfficialComparisonCitations(task,newer.result.text),true);
 assert.equal(resolveLearningExampleSuccess(older),false);
 assert.equal(resolveLearningExampleSuccess(newer),true);
});
test("quarantine a false positive without inflating maturity or retaining stale examples",()=>{
 const existing={skillId:"comparison-workflow",name:"Document comparison",category:"legal",workflow:["Fetch official source"],examples:[older],occurrences:1,successRate:1,confidence:0.7,matchScore:1,requiredTools:["web_fetch"]};
 const incoming={...existing,examples:[newer],occurrences:1};
 const merged=mergeLearningCandidates({existing,incoming,similarity:0.9});
 assert.equal(merged.examples.length,1);
 assert.equal(merged.examples[0].traceId,"new");
 assert.equal(merged.occurrences,1);
 assert.equal(merged.successCount,1);
 assert.equal(merged.failureCount,0);
});
test("general tasks unrelated to official citations remain eligible",()=>{
 assert.equal(hasRequestedOfficialComparisonCitations("Сравни две архитектуры сервисов","Без ссылок"),true);
});
