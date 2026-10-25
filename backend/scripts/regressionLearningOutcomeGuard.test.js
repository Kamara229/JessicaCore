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

test("optimistic completion without a verified result never enters Learning Queue",()=>{
    for (const trace of [
        {id:"trace-no-result",status:"COMPLETED",stats:{completed:1}},
        {id:"trace-empty-results",status:"COMPLETED",stats:{completed:1},
            result:{status:"COMPLETED",success:true,results:[]}},
        {id:"trace-unvalidated",status:"COMPLETED",stats:{completed:1},
            result:{status:"COMPLETED",success:true,verified:true}},
        {id:"trace-unconfirmed",status:"COMPLETED",stats:{completed:1},
            result:{status:"COMPLETED",verified:true,validation:{valid:true}}}
    ]) {
        assert.equal(shouldAnalyzeLearningTrace(trace),false,trace.id);
    }
});

import {inspectAppliedSkill} from "../core/trace/traceSkillApplication.js";
import {buildSkillTraceView} from "../core/learning/multiSkill/skillTraceView.js";

test("selected Skill does not count as applied without matching tool execution",()=>{
    const skill={id:"license-patent-clause-comparison"};
    const context={
        experience:{used:true,skills:[skill]},
        plan:{experience:{used:true,skills:[skill]},
            steps:[{id:"source",tool:"web_fetch",experienceUsed:true}]},
        runResult:{success:true,results:[]}
    };
    assert.equal(inspectAppliedSkill(context).applied,false);
    context.runResult.results.push({id:"source",tool:"web_fetch",success:true});
    assert.equal(inspectAppliedSkill(context).applied,true);
    assert.deepEqual(inspectAppliedSkill(context).skillIds,[skill.id]);
    context.runResult.results[0].success=false;
    assert.equal(inspectAppliedSkill(context).applied,false);
});

test("positive Skill reinforcement requires both applied child and verified outcome",()=>{
    const skill={id:"license-patent-clause-comparison"};
    const child={
        task:"Read a primary legal source",status:"COMPLETED",
        success:true,verified:true,
        validation:{valid:true,degraded:false,outcomeType:"result"},
        experience:{used:true,applied:true,appliedSkillIds:[skill.id],skills:[skill]}
    };
    const trace={
        id:"applied-source",task:"Study patent grants",status:"COMPLETED",
        statistics:{completed:1},
        experienceUsage:{used:true,applied:true,appliedSkillIds:[skill.id],skills:[skill]},
        result:{status:"COMPLETED",success:true,results:[child]}
    };
    const proven=buildSkillTraceView(trace,skill);
    assert.ok(proven);
    assert.equal(proven.experienceUsage.applied,true);
    assert.equal(shouldAnalyzeLearningTrace(proven),true);
    const selectedOnly={...trace,result:{...trace.result,results:[
        {...child,experience:{...child.experience,applied:false}}
    ]}};
    assert.equal(buildSkillTraceView(selectedOnly,skill),null);
    const unverified={...trace,result:{...trace.result,results:[
        {...child,verified:false}
    ]}};
    const invalid=buildSkillTraceView(unverified,skill);
    assert.ok(invalid);
    assert.equal(shouldAnalyzeLearningTrace(invalid),false);
});
