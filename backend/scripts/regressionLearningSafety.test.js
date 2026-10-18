/*
 * Regression checks for autonomous Skill learning.
 * No credentials, network calls, or external dependencies.
 */
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
