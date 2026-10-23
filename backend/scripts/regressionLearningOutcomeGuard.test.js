import test from "node:test";
import assert from "node:assert/strict";
import {hasNegativeLearningEvidence} from "../core/learning/trigger/traceEvidenceGuard.js";
import {shouldAnalyzeLearningTrace} from "../core/learning/trigger/triggerEligibility.js";

const success=()=>({
    id:"trace-verified",
    task:"Calculate and verify a date interval",
    status:"COMPLETED",
    stats:{completed:1,failed:0},
    result:{
        status:"COMPLETED",success:true,verified:true,
        validation:{valid:true,degraded:false,outcomeType:"result"}
    }
});

test("verified task remains eligible for automatic Experience learning",()=>{
    assert.equal(hasNegativeLearningEvidence(success()),false);
    assert.equal(shouldAnalyzeLearningTrace(success()),true);
});

test("completion counters never upgrade no_verified_result into positive learning",()=>{
    for (const variant of [
        {result:{status:"NO_VERIFIED_RESULT",success:true}},
        {result:{status:"COMPLETED",success:true,verified:true,
            validation:{valid:true,outcomeType:"no_verified_result"}}},
        {validation:{valid:true,outcomeType:"no_verified_result"}},
        {result:{status:"COMPLETED",success:true,verified:true,
            validation:{valid:true,degraded:true}}}
    ]) {
        const trace={...success(),...variant};
        assert.equal(hasNegativeLearningEvidence(trace),true);
        assert.equal(shouldAnalyzeLearningTrace(trace),false);
    }
});

test("failed/partial/needs-clarification outcome cannot be hidden by successful stats",()=>{
    for(const status of ["FAILED","PARTIAL","NEEDS_CLARIFICATION","CANCELLED"]){
        assert.equal(shouldAnalyzeLearningTrace({...success(),status}),false,status);
    }
    assert.equal(shouldAnalyzeLearningTrace({...success(),success:false}),false);
});

test("unverified child verdict blocks otherwise successful multi-step trace",()=>{
    const trace={...success(),result:{
        status:"COMPLETED",success:true,results:[
            {status:"COMPLETED",success:true,verified:true,
                validation:{valid:true,outcomeType:"result"}},
            {status:"COMPLETED",success:true,verified:true,
                validation:{valid:true,outcomeType:"no_verified_result"}}
        ]
    }};
    assert.equal(hasNegativeLearningEvidence(trace),true);
    assert.equal(shouldAnalyzeLearningTrace(trace),false);
    trace.result.results[1].validation.outcomeType="result";
    assert.equal(hasNegativeLearningEvidence(trace),false);
    assert.equal(shouldAnalyzeLearningTrace(trace),true);
});

test("unknown validation is not invented as positive evidence",()=>{
    const trace={...success(),result:{status:"COMPLETED",success:true}};
    assert.equal(hasNegativeLearningEvidence(trace),false);
    assert.equal(shouldAnalyzeLearningTrace(trace),false);
});
