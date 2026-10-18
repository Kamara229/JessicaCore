/*
 * Regression checks for autonomous Skill learning.
 * No credentials, network calls, or external dependencies.
 */
import "../tools/dateDiffTool.js";
import test from "node:test";
import assert from "node:assert/strict";
import { executeReleaseFact } from "../tools/releaseFact/releaseFactExecutor.js";
import { classifyLearningOutcome } from "../core/learning/learningOutcomePolicy.js";
import { resolveLearningExampleSuccess } from "../core/learning/exampleSuccess.js";
import { inspectSkillEvidence } from "../core/learning/quality/inspectorEvidence.js";
import { buildSkillTraceView } from "../core/learning/multiSkill/skillTraceView.js";
import { evaluateSkillEvolution } from "../core/learning/quality/skillEvolutionGuard.js";

const official = "https://www.python.org/downloads/";
const releases = [
    "Python releases by version number",
    "Python 3.15.0 Oct. 9, 2026 Download",
    "Python 3.14.8 Sept. 30, 2026 Download",
    "Python 3.14.7 Aug. 5, 2026 Download",
    "Python 3.14.6 June 10, 2026 Download"
].join("\n");

function extract(content) {
    return executeReleaseFact({
        content,
        sourceUrl: official,
        sourceTitle: "Download Python | Python.org",
        subject: "Python"
    });
}

test("latest stable Python release includes its OWN date", () => {
    const result = extract(releases);
    assert.equal(result.success, true);
    assert.equal(result.data.version, "3.15.0");
    assert.equal(result.data.releaseDate, "2026-10-09");
});

test("planned and prerelease rows are excluded", () => {
    const result = extract("Python 3.16 pre-release 2027-10-06 (planned)\n" + releases);
    assert.equal(result.data.version, "3.15.0");
});

test("date from next version cannot be borrowed by older heading", () => {
    const result = extract("Python 3.14.7\nPython 3.14.6 June 10, 2026");
    assert.equal(result.success, true);
    assert.equal(result.data.version, "3.14.6");
    assert.equal(result.data.releaseDate, "2026-06-10");
});

const good = {
    success: true,
    result: {
        status: "COMPLETED",
        success: true,
        verified: true,
        text: "Confirmed",
        validation: { valid: true, degraded: false }
    }
};
const invalid = {
    success: false,
    result: {
        status: "COMPLETED",
        success: true,
        verified: false,
        text: null,
        validation: { valid: false, degraded: false }
    }
};

test("failed child execution is never positive learning", () => {
    assert.equal(resolveLearningExampleSuccess(invalid), false);
    assert.equal(classifyLearningOutcome({result: invalid.result}).positive, false);
});

test("newly invalid evidence cannot be hidden by old successful examples", () => {
    const result = inspectSkillEvidence(
        {examples: [good, invalid]},
        {action: "SKILL_IMPROVEMENT", analysis: {evolution: {newEvidenceCount: 1}}}
    );
    assert.ok(result.errors.length > 0);
});

test("successful parent without per-skill result cannot reinforce missing child", () => {
    const skill = {id: "date-difference-calculation"};
    const source = {id: "official-source-verification"};
    const trace = {
        task: "Compound task",
        result: {
            results: [{
                task: "Source lookup",
                experience: {used: true, skills: [source]}
            }]
        }
    };
    assert.equal(buildSkillTraceView(trace, skill), null);
});

test("READY skill with unchanged knowledge is reinforced without version bump", () => {
    const result = evaluateSkillEvolution({
        action: "SKILL_IMPROVEMENT",
        analysis: {
            evolution: {
                baseMaturityLevel: "READY",
                proposedMaturityLevel: "READY",
                knowledgeChanged: false,
                maturityChanged: false,
                newEvidenceCount: 1
            }
        }
    });
    assert.equal(result.mode, "REINFORCEMENT_ONLY");
});

/* Regression: distinguish branch versions from exact releases. */
import { scoreReleaseSourcePreference } from "../core/source/releaseSourcePreference.js";
import { hasVerifiedDateSkillEvidence } from "../core/planner/youngDateSkill.js";
import { tryCreateNativeSkillPlan } from "../core/planner/nativeSkillPlanner.js";

test("Python branch status must never be claimed as exact release", () => {
    const result = executeReleaseFact({
        content: "Status of Python versions\nBranch | Status | First release\nPython 3.15 | bugfix | 2026-10-09",
        sourceUrl: "https://devguide.python.org/versions/",
        sourceTitle: "Status of Python versions",
        subject: "latest stable Python release date"
    });
    assert.equal(result.success, false);
    assert.equal(result.data?.needsExactReleaseSource, true);
});

test("official Python downloads index outranks branch status page", () => {
    const task = "Найди последний стабильный релиз Python";
    assert.ok(
        scoreReleaseSourcePreference(task, {url: "https://www.python.org/downloads/"})
        > scoreReleaseSourcePreference(task, {url: "https://devguide.python.org/versions/"})
    );
});

const verifiedYoungSkill = {
    id: "date-difference-calculation",
    enabled: true,
    learning: {maturityLevel:"NEW", confidence:0.76, successCount:1},
    executionProgram: {mode:"native",strategy:"date_difference",evidenceMode:"none"},
    examples: [{
        success:true,
        result:{
            success:true,verified:true,status:"COMPLETED",
            validation:{valid:true,degraded:false},
            executionMeta:{usedTools:["date_diff"]}
        },
        executedTools:["date_diff"]
    }]
};

test("verified NEW date Skill can be reused instead of anonymous native capability", () => {
    assert.equal(hasVerifiedDateSkillEvidence(verifiedYoungSkill), true);
    const plan = tryCreateNativeSkillPlan({
        task:"рассчитай, сколько полных дней прошло до 18 октября 2026 года",
        dependencyContext:"Дата выпуска 2026-10-09",
        executionExperience: {skills:[verifiedYoungSkill]},
        planningContext:{}
    });
    assert.equal(plan.handled, true);
    assert.equal(plan.source,"experience:date-difference-calculation");
    assert.equal(plan.plan.experience.used,true);
});

test("unverified NEW date Skill cannot be reinforced as already applied", () => {
    const corrupt={
        ...verifiedYoungSkill,
        examples:[{
            ...verifiedYoungSkill.examples[0],
            result:{...verifiedYoungSkill.examples[0].result, verified:false}
        }]
    };
    assert.equal(hasVerifiedDateSkillEvidence(corrupt), false);
    const plan = tryCreateNativeSkillPlan({
        task:"рассчитай, сколько полных дней прошло до 18 октября 2026 года",
        dependencyContext:"Дата выпуска 2026-10-09",
        executionExperience:{skills:[corrupt]},
        planningContext:{}
    });
    assert.equal(plan.handled,true);
    assert.equal(plan.source,"native-date-diff");
});

import { buildPatternEvidence } from "../core/learning/patternExtraction/patternEvidence.js";
import { hasVerifiedPatternEvidence } from "../core/learning/patternExtraction/patternEvidencePolicy.js";

const verifiedFetch = {
    success:true,verified:true,status:"COMPLETED",
    validation:{valid:true, degraded:false, outcomeType:"result"},
    answer:{text:"Read both verified primary sources"}
};
const verifiedComparison = {
    success:true,verified:true,status:"COMPLETED",
    validation:{valid:true, degraded:false, outcomeType:"result"},
    answer:{text:"ShareAlike is mandatory when distributing adapted material."}
};
const ccTrace = {
    task:"Compare the CC BY and CC BY-SA licenses",
    status:"COMPLETED",
    result:{
        success:true,status:"COMPLETED",
        results:[verifiedFetch,verifiedComparison]
    },
    executedTools:["web_search","web_fetch"]
};

test("autonomous discovery inherits verified answer from complex child", () => {
    const evidence=buildPatternEvidence({trace:ccTrace});
    assert.equal(evidence.answer,verifiedComparison.answer.text);
    assert.equal(evidence.validation.valid,true);
    assert.equal(evidence.result.verified,true);
    assert.equal(evidence.result.success,true);
});

test("failed prerequisite cannot become positive discovery evidence", () => {
    const evidence=buildPatternEvidence({trace:{
        ...ccTrace,
        result:{...ccTrace.result,results:[
            {...verifiedFetch, verified:false},verifiedComparison
        ]}
    }});
    assert.equal(evidence.result.verified,false);
    assert.equal(evidence.validation.valid,false);
});

test("legacy empty pattern evidence cannot pass quality gate", () => {
    const evidence={
        task:"Compare two licenses",
        answer:"",
        result:{success:true,status:"COMPLETED"},
        validation:{valid:false},
        tools:["web_search","web_fetch"]
    };
    assert.equal(hasVerifiedPatternEvidence(evidence),false);
});

test("verified complex discovery evidence passes quality gate", () => {
    const evidence=buildPatternEvidence({trace:ccTrace});
    assert.equal(hasVerifiedPatternEvidence(evidence),true);
});
