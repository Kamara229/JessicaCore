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

import { matchExperiencePattern } from "../core/learning/experiencePatternMatcher.js";

test("one generic official-source keyword must not create an existing seed Skill", () => {
    const task = "Сопоставь условия лицензий и приведи ссылку на источник.";
    assert.equal(matchExperiencePattern(task), null);
});

test("multiple discriminating keywords can match a genuine known pattern", () => {
    const task = "Найди телефон, email и почту поддержки; проверь контакт и адрес.";
    const matched = matchExperiencePattern(task);
    assert.equal(matched?.pattern?.id, "company-contact-finder");
    assert.ok(matched.matchScore >= 0.5);
});

import {
    resolveReferencedSearchResult,
    scopeCandidatesToSearchQuery,
    selectionContextForSearch
} from "../core/taskRunner/argumentResolver/sourceQueryScope.js";

const makeSearch=(id,query,results)=>({
    id, tool:"web_search",success:true,arguments:{query},data:{results}
});
const bothLicenses=[
    {title:"Apache License, Version 2.0",url:"https://opensource.org/license/apache-2.0"},
    {title:"The MIT License",url:"https://opensource.org/license/mit"},
    {title:"Open Source Licenses by Category",url:"https://opensource.org/licenses-old/category"},
    {title:"Apache Public Licenses",url:"https://opensource.org/licenses-old/apachepl-php"}
];

test("each fetch reference resolves its own search step rather than most recent",()=>{
    const mit=makeSearch("search_mit","MIT License site:opensource.org",bothLicenses);
    const apache=makeSearch("search_apache","Apache License 2.0 site:opensource.org",bothLicenses);
    assert.equal(resolveReferencedSearchResult({$from:"search_mit"},[mit,apache]),mit);
    assert.equal(resolveReferencedSearchResult({$from:"search_apache"},[mit,apache]),apache);
    assert.equal(resolveReferencedSearchResult({$from:"unknown"},[mit,apache]),null);
    assert.equal(selectionContextForSearch({task:"compare MIT and Apache"},mit).task,mit.arguments.query);
});

test("MIT fetch cannot select Apache page or generic index",()=>{
    const scoped=scopeCandidatesToSearchQuery("MIT License site:opensource.org",bothLicenses);
    assert.deepEqual(scoped.map(x=>x.url),["https://opensource.org/license/mit"]);
});

test("Apache 2.0 fetch cannot select MIT or obsolete generic Apache URLs",()=>{
    const scoped=scopeCandidatesToSearchQuery("Apache License 2.0 site:opensource.org",bothLicenses);
    assert.deepEqual(scoped.map(x=>x.url),["https://opensource.org/license/apache-2.0"]);
});

test("no matching named license yields empty candidates, not an unrelated source",()=>{
    assert.deepEqual(scopeCandidatesToSearchQuery("MIT License site:opensource.org",[bothLicenses[0],bothLicenses[2]]),[]);
});

import { requestedCreativeCommonsVariant, filterCreativeCommonsSources }
    from "../core/taskRunner/argumentResolver/creativeCommonsScope.js";

const ccResults=[
    {title:"Deed - Attribution 4.0 International",url:"https://creativecommons.org/licenses/by/4.0/deed.en"},
    {title:"Deed - Attribution-ShareAlike 4.0 International",url:"https://creativecommons.org/licenses/by-sa/4.0/deed.en"},
    {title:"Deed - Attribution-ShareAlike 3.0 Unported",url:"https://creativecommons.org/licenses/by-sa/3.0/deed.en"},
    {title:"Creative Commons licences",url:"https://creativecommons.org/licenses/"}
];

test("Creative Commons BY 4.0 fetch uses only exact BY 4.0",()=>{
    const query="Creative Commons CC BY 4.0 license description official site";
    assert.deepEqual(requestedCreativeCommonsVariant(query),{family:"by",version:"4.0"});
    assert.deepEqual(scopeCandidatesToSearchQuery(query,ccResults).map(x=>x.url),
        ["https://creativecommons.org/licenses/by/4.0/deed.en"]);
});

test("Creative Commons BY-SA 4.0 excludes both CC BY 4.0 and 3.0",()=>{
    const query="Creative Commons BY-SA 4.0 license deed official site";
    assert.deepEqual(requestedCreativeCommonsVariant(query),{family:"by-sa",version:"4.0"});
    assert.deepEqual(filterCreativeCommonsSources(query,ccResults).map(x=>x.url),
        ["https://creativecommons.org/licenses/by-sa/4.0/deed.en"]);
});

test("Creative Commons version mismatch fails source lookup instead of trusting a wrong licence",()=>{
    const query="Creative Commons CC BY-SA 4.0 license";
    assert.deepEqual(scopeCandidatesToSearchQuery(query,[ccResults[0],ccResults[2]]),[]);
});

import { deriveVerifiedDiscoveryMetrics } from "../core/learning/confidence/verifiedDiscoveryMetrics.js";
import { buildPatternEvidence } from "../core/learning/patternExtraction/patternEvidence.js";
import { buildNewSkillCandidate } from "../core/learning/experienceCandidateBuilder.js";
import { mergeLearningCandidates } from "../core/learning/candidateMemory/candidateMerge.js";

const verifiedDiscoveryPattern={
    id:"two-document-condition-comparison",
    name:"Document terms comparison",
    category:"document-analysis",
    description:"Retrieve original versions and compare clauses against explicit criteria",
    workflow:["Fetch each original document","Verify versions","Compare clauses with citations"],
    triggerPatterns:["compare two documents","сравни условия двух документов"],
    validationRules:["Each conclusion cites a fetched source"],
    constraints:["Do not invent clauses"],
    requiredTools:["web_fetch"]
};

function makeVerifiedDiscoveryTrace(id) {
    return {
        id,task:"Сравни два документа",
        status:"COMPLETED",success:true,
        executedTools:["web_fetch"],
        result:{
            status:"COMPLETED",success:true,verified:true,
            answer:{text:"Подтверждено сопоставление условий двух исходных документов."},
            validation:{valid:true,degraded:false,outcomeType:"result"}
        }
    };
}

test("discovery cannot promote a Skill from unverified or degraded evidence",()=>{
    const trace=makeVerifiedDiscoveryTrace("trace-good");
    const evidence=buildPatternEvidence({trace,metrics:{successRate:0,occurrences:99}});
    assert.equal(deriveVerifiedDiscoveryMetrics(evidence)?.confidence,0.76);
    const invalid={...evidence,result:{...evidence.result,verified:false}};
    assert.equal(deriveVerifiedDiscoveryMetrics(invalid),null);
    const degraded={...evidence,validation:{...evidence.validation,degraded:true}};
    assert.equal(deriveVerifiedDiscoveryMetrics(degraded),null);
});

test("first verified discovery stays a candidate; independent second success raises confidence",()=>{
    function candidate(id) {
        const trace=makeVerifiedDiscoveryTrace(id);
        const evidence=buildPatternEvidence({trace,metrics:{occurrences:999,successRate:0}});
        const m=deriveVerifiedDiscoveryMetrics(evidence);
        return buildNewSkillCandidate({
            pattern:verifiedDiscoveryPattern,
            trace,
            confidence:m.confidence,
            maturity:m.maturity,
            occurrences:m.occurrences
        });
    }
    const first=candidate("trace-1");
    assert.equal(first.examples.length,1);
    assert.equal(first.examples[0].result.verified,true);
    assert.equal(first.examples[0].result.validation.valid,true);
    assert.equal(first.successCount,1);
    assert.equal(first.successRate,1);
    assert.equal(first.confidence,0.76);
    const second=candidate("trace-2");
    const merged=mergeLearningCandidates({existing:first,incoming:second,exactMatch:true});
    assert.equal(merged.examples.length,2);
    assert.equal(merged.successCount,2);
    assert.equal(merged.failureCount,0);
    assert.equal(merged.successRate,1);
    assert.equal(merged.occurrences,2);
    assert.equal(merged.confidence,0.82);
    const duplicate=mergeLearningCandidates({existing:merged,incoming:second,exactMatch:true});
    assert.equal(duplicate.occurrences,2);
    assert.equal(duplicate.confidence,0.82);
});
