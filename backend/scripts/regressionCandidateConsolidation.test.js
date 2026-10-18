import test from "node:test";
import assert from "node:assert/strict";
import { planCandidateConsolidation } from "../core/learning/candidateMemory/candidateConsolidator.js";
import { mergeLearningCandidates } from "../core/learning/candidateMemory/candidateMerge.js";
import { candidateProcedureFamily } from "../core/learning/candidateMemory/candidateFamily.js";

const comparisonA = {
    skillId: "document-comparison-and-summary",
    name: "Document Comparison and Summary",
    category: "Information Retrieval & Analysis",
    workflow: [
        "Identify the two target documents and their URLs or identifiers.",
        "Fetch each document using a web or file retrieval tool.",
        "Define comparison criteria.",
        "Compile the findings into a structured comparison table.",
        "Reference the documents to support each finding."
    ],
    triggerPatterns: ["compare two documents", "compare two licenses"],
    examples: [{task:"compare BSD 2 and 3",traceId:"verified-trace-one",
        result:{status:"COMPLETED",success:true}}],
    occurrences: 1,
    confidence: 0.76
};
const comparisonB = {
    skillId:"compare-and-summarize-document-differences",
    name:"Compare and Summarize Document Differences",
    category:"Document Analysis",
    workflow:[
        "Retrieve each document independently (e.g. via web_fetch).",
        "Extract comparison criteria and create a comparison table.",
        "Reference the source documents to support the answer."
    ],
    triggerPatterns:["Compare two documents","Analyze differences"],
    examples:[{task:"compare CC BY and BY-SA",traceId:"verified-trace-two",
        result:{status:"COMPLETED",success:true}}],
    occurrences:1,
    confidence:0.76
};

test("autonomously detects the two independently created live comparison candidates", () => {
    assert.equal(candidateProcedureFamily(comparisonA),"two-original-documents-comparison");
    assert.equal(candidateProcedureFamily(comparisonB),"two-original-documents-comparison");
    const plan=planCandidateConsolidation([
        {id:"old",status:"ACTIVE",createdAt:"2026-10-18T14:43:00Z",candidate:comparisonA},
        {id:"new",status:"ACTIVE",createdAt:"2026-10-18T15:33:00Z",candidate:comparisonB}
    ]);
    assert.equal(plan.length,1);
    assert.equal(plan[0].canonical.id,"old");
    assert.equal(plan[0].duplicate.id,"new");
    assert.ok(plan[0].similarity >= .75);
});

test("candidate merge preserves stable identity and both traces", () => {
    const merged=mergeLearningCandidates({
        existing:comparisonA,incoming:comparisonB,similarity:.75
    });
    assert.equal(merged.skillId,comparisonA.skillId);
    assert.equal(merged.examples.length,2);
    assert.equal(merged.occurrences,2);
    assert.ok(merged.examples.some(e=>e.traceId==="verified-trace-one"));
    assert.ok(merged.examples.some(e=>e.traceId==="verified-trace-two"));
});

test("prevents unrelated or inactive candidates from cross-category merging", () => {
    const unrelated={...comparisonB,
        name:"Official website verification",
        workflow:["Check source URL and retrieve website"],
        triggerPatterns:["find official site"]
    };
    assert.equal(candidateProcedureFamily(unrelated),null);
    const entries=[
        {id:"old",status:"ACTIVE",createdAt:"2026-10-18T14:43:00Z",candidate:comparisonA},
        {id:"new",status:"ACTIVE",createdAt:"2026-10-18T15:33:00Z",candidate:unrelated},
        {id:"inactive",status:"MERGED",createdAt:"2026-10-18T15:33:00Z",candidate:comparisonB}
    ];
    assert.deepEqual(planCandidateConsolidation(entries),[]);
});

test("one ACTIVE candidate requires no consolidation", () => {
    assert.equal(planCandidateConsolidation([]).length,0);
    assert.equal(planCandidateConsolidation([{
        id:"old",status:"ACTIVE",candidate:comparisonA
    }]).length,0);
});
