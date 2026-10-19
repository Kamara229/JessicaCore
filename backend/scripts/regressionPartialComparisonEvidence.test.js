import test from "node:test";
import assert from "node:assert/strict";
import { buildPartialComparisonEvidence } from "../core/composer/partialComparisonEvidence.js";
import { executeAnswerValidation } from "../core/execution/stepExecution/answerValidationCoordinator.js";
import { classifyLearningOutcome } from "../core/learning/learningOutcomePolicy.js";
import { buildSingleTaskResponse } from "../core/response/singleTaskResponse.js";

const apache="https://www.apache.org/licenses/LICENSE-2.0.html";
const bsd="https://opensource.org/license/bsd-3-clause";
const task="Сравни использование, распространение, уведомления и патенты в Apache и BSD.";
const fetches={results:[
    {tool:"web_fetch",success:true,data:{url:apache,content:"perpetual worldwide non exclusive patent license. You may reproduce and distribute copies."}},
    {tool:"web_fetch",success:true,data:{url:bsd,content:"Redistribution and use in source and binary forms. Redistributions of source code must retain notices."}}
]};
const claims=[
    {supported:true,sourceUrl:apache,evidence:"perpetual worldwide non exclusive patent license"},
    {supported:true,sourceUrl:apache,evidence:"You may reproduce and distribute copies"},
    {supported:true,sourceUrl:bsd,evidence:"Redistribution and use in source and binary forms"},
    {supported:true,sourceUrl:bsd,evidence:"Redistributions of source code must retain notices"},
    {supported:false,sourceUrl:bsd,evidence:"No patent provisions",claim:"BSD lacks patents"}
];

test("partial response contains only grounded attributed excerpts; never unsupported conclusions",()=>{
    const p=buildPartialComparisonEvidence(task,{evidence:{mode:"source_content"}},fetches,claims);
    assert.ok(p);
    assert.match(p.text,/Частичный результат/);
    assert.ok(p.text.includes(apache));
    assert.ok(p.text.includes(bsd));
    assert.match(p.text,/perpetual worldwide/);
    assert.doesNotMatch(p.text,/BSD lacks patents|No patent provisions/);
    assert.deepEqual(p.quoteCounts,[2,2]);
});

test("partial fallback refuses one-sided or weakly sourced tasks",()=>{
    const plan={evidence:{mode:"source_content"}};
    assert.equal(buildPartialComparisonEvidence(task,plan,fetches,claims.filter(x=>x.sourceUrl===apache)),null);
    assert.equal(buildPartialComparisonEvidence("Расскажи документ",plan,fetches,claims),null);
    assert.equal(buildPartialComparisonEvidence(task,{evidence:{mode:"none"}},fetches,claims),null);
});

test("one failed repair creates degraded source-excerpt result, never positive learning",async()=>{
    const context={task,plan:{evidence:{mode:"source_content"}},runResult:fetches,
        metadata:{},answerResult:null,validationResult:null};
    const bad={success:false,failure:{category:"presentation",failureScope:"presentation",
        reason:"Неподтвержденные утверждения",
        validation:{valid:false,claims}}};
    const stages=[];
    const result=await executeAnswerValidation(context,async (_,stage)=>{
        stages.push(stage);
        if(stage==="composer" || stage==="composer-repair"){
            context.answerResult={success:true,text:"Недостаточно подтвержденный ответ"};
            return {success:true};
        }
        context.validationResult={valid:false};
        return bad;
    });
    assert.equal(result.success,true);
    assert.deepEqual(stages,["composer","validator","composer-repair","validator-repair"]);
    assert.equal(context.validationResult.degraded,true);
    assert.equal(context.validationResult.outcomeType,"no_verified_result");
    assert.match(context.answerResult.text,/Частичный результат/);
    const learning=classifyLearningOutcome({
        result:{status:"COMPLETED",success:true,verified:false,
            validation:context.validationResult}
    });
    assert.equal(learning.positive,false);
});


test("validator repair instructions are not shown verbatim to the user",()=>{
    const api=buildSingleTaskResponse({status:"FAILED",failure:{
        failureType:"answer-validation-error",stage:"validator",
        reason:"Не подтверждено утверждений: 3. При повторной генерации исключи эти тезисы"
    }},null);
    assert.equal(api.success,false);
    assert.match(api.text,/не смогла подтвердить все выводы/);
    assert.doesNotMatch(api.text,/При повторной генерации|Не подтверждено утверждений/);
});
