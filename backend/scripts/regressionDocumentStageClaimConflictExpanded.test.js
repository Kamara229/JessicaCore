import test from "node:test";
import assert from "node:assert/strict";
import {evaluateStageClaimConflictFixtures}
 from "../core/files/documentStageClaimConflictQualityFixtures.js";
import {expandedConflictBenchmarkCases}
 from "./fixtures/documentStageClaimConflictExpandedCases.js";

test("22 hand-labelled synthetic cases expose false positives and misses",()=>{
    const benchmark=evaluateStageClaimConflictFixtures(
        expandedConflictBenchmarkCases);
    assert.equal(benchmark?.casesEvaluated,22);
    assert.equal(benchmark.fixtureTruePositives,13);
    assert.equal(benchmark.fixtureFalsePositives,6);
    assert.equal(benchmark.fixtureMissedPairs,0);
    assert.equal(benchmark.fixtureReviewTierMismatches,0);
    assert.equal(benchmark.fixturePrecision,13/19);
    assert.equal(benchmark.fixtureRecall,1);
    assert.equal(benchmark.labelOrigin,
        "explicitly-annotated-synthetic-fixtures");
    assert.equal(benchmark.errors.length,6);
    assert.equal(benchmark.realWorldAccuracyMeasured,false);
    assert.equal(benchmark.semanticConflictVerified,false);
});

test("explicitly cover dates conditions exceptions paraphrases and controls",()=>{
    assert.equal(new Set(expandedConflictBenchmarkCases.map(x=>x.id)).size,
        expandedConflictBenchmarkCases.length);
    assert.deepEqual(
        [...new Set(expandedConflictBenchmarkCases.map(x=>x.category))].sort(),
        ["conditions","control","dates","exceptions","paraphrases",
            "polarity"]);
    const ids=new Set(expandedConflictBenchmarkCases.map(x=>x.id));
    for(const needed of ["different-dates","same-date",
        "different-conditions","one-sided-condition","exception",
        "review-qualifier","force-paraphrase","modal-paraphrase",
        "morphological-paraphrase","same-window"]){
        assert.ok(ids.has(needed));
    }
});

test("known synthetic errors are retained, not misrepresented as accuracy",()=>{
    const result=evaluateStageClaimConflictFixtures(
        expandedConflictBenchmarkCases);
    const errors=(type)=>result.errors.filter(x=>x.type===type)
        .map(x=>x.caseId);
    assert.deepEqual(errors("fixture-false-positive"),[
        "different-dates","different-conditions","exception",
        "one-sided-condition","date-on-one-side","date-and-exception"]);
    assert.deepEqual(errors("fixture-missed-pair"),[]);
    assert.equal(result.errors.length,6);
});

test("every synthetic claim carries traceable source location and evidence",()=>{
    for(const sample of expandedConflictBenchmarkCases){
        assert.equal(sample.claims.length,2);
        for(const claim of sample.claims){
            assert.ok(claim.claim.includes("[О"+
                claim.windowNumber+":Ф1:Ф1]"));
            assert.ok(claim.evidence.startsWith("  Основание: «"));
            assert.ok(claim.startOffset<claim.endOffset);
            assert.ok(Number.isSafeInteger(claim.sectionIndex));
        }
    }
});
