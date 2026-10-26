import test from "node:test";
import assert from "node:assert/strict";
import {evaluateStageClaimConflictFixtures}
 from "../core/files/documentStageClaimConflictQualityFixtures.js";
import {auditStageClaimConflictTriage}
 from "../core/files/documentStageClaimConflictQualityAudit.js";
import {groupStageClaimSelections}
 from "../core/files/documentStageClaimSelectionGrouping.js";
import {findStageClaimConflictCandidates}
 from "../core/files/documentStageClaimConflictCandidates.js";
import {triageStageClaimConflictCandidates}
 from "../core/files/documentStageClaimConflictTriage.js";

const entry=(window,statement,evidence)=>({
    stageNumber:1,windowNumber:window,sectionIndex:1,
    startOffset:window*100,endOffset:window*100+50,
    claim:"• Позиция автора: "+statement+" [О"+window+":Ф1:Ф1]",
    evidence:"  Основание: «"+evidence+"»"
});
const pair=(tier="primary-review")=>[{
    affirmativeWindow:1,negativeWindow:2,reviewTier:tier
}];

const samples=[
    {id:"direct-polarity",claims:[
        entry(1,"Договор действует","Договор действует"),
        entry(2,"Договор не действует","Договор не действует")
    ],expectedPairs:pair()},
    {id:"separate-periods",claims:[
        entry(1,"Договор действует","По состоянию на 01.01.2024 договор действует"),
        entry(2,"Договор не действует","По состоянию на 02.01.2024 договор не действует")
    ],expectedPairs:[]},
    {id:"different-explicit-conditions",claims:[
        entry(1,"Договор действует","Договор действует при наличии решения"),
        entry(2,"Договор не действует","Договор не действует при наличии согласия")
    ],expectedPairs:[]},
    {id:"lexical-paraphrase-miss",claims:[
        entry(1,"Договор вступил в силу","Договор вступил в силу"),
        entry(2,"Договор не вступил в силу","Договор не вступил в силу")
    ],expectedPairs:pair()},
    {id:"same-polarity",claims:[
        entry(1,"Договор действует","Договор действует"),
        entry(2,"Договор действует","Договор действует")
    ],expectedPairs:[]},
    {id:"same-source-window",claims:[
        entry(1,"Договор действует","Договор действует"),
        entry(1,"Договор не действует","Договор не действует")
    ],expectedPairs:[]},
    {id:"same-date",claims:[
        entry(1,"Договор действует","На 01.02.2024 договор действует"),
        entry(2,"Договор не действует","На 01.02.2024 договор не действует")
    ],expectedPairs:pair()},
    {id:"qualified-uncertain-pair",claims:[
        entry(1,"Договор действует","Договор действует"),
        entry(2,"Договор не действует","Договор не действует, если наступило условие")
    ],expectedPairs:pair("context-qualified-review")}
];

test("benchmark reports fixture false positives and missed pairs without claiming science",()=>{
    const quality=evaluateStageClaimConflictFixtures(samples);
    assert.equal(quality.casesEvaluated,8);
    assert.equal(quality.fixtureTruePositives,3);
    assert.equal(quality.fixtureFalsePositives,2);
    assert.equal(quality.fixtureMissedPairs,1);
    assert.equal(quality.fixtureReviewTierMismatches,0);
    assert.equal(quality.fixturePrecision,3/5);
    assert.equal(quality.fixtureRecall,3/4);
    assert.equal(quality.realWorldAccuracyMeasured,false);
    assert.equal(quality.semanticConflictVerified,false);
    assert.ok(quality.errors.some(e=>e.caseId==="lexical-paraphrase-miss"
        &&e.type==="fixture-missed-pair"));
    assert.ok(quality.errors.some(e=>e.caseId==="separate-periods"
        &&e.type==="fixture-false-positive"));
});

test("audit verifies every original candidate citation after prioritization",()=>{
    const grouped=groupStageClaimSelections([
        samples[0].claims[0],samples[1].claims[1],
        entry(3,"Договор не действует","Договор не действует")
    ]);
    const found=findStageClaimConflictCandidates(grouped);
    const ranked=triageStageClaimConflictCandidates(found);
    const audit=auditStageClaimConflictTriage(found,ranked);
    assert.equal(audit?.conflictPairsAudited,2);
    assert.equal(audit.conflictCitationOccurrencesPreserved,4);
    assert.equal(audit.conflictReviewCandidatesLost,0);
    assert.equal(audit.conflictTriageRetentionVerified,true);
    assert.equal(audit.realWorldMissedConflictsMeasured,false);
});

test("audit fails closed if a source is altered, lost or review order breaks",()=>{
    const grouped=groupStageClaimSelections([
        samples[0].claims[0],samples[1].claims[1],
        entry(3,"Договор не действует","Договор не действует")
    ]);
    const found=findStageClaimConflictCandidates(grouped);
    const ranked=triageStageClaimConflictCandidates(found);
    for(const modify of [
        r=>r.candidates.pop(),
        r=>r.candidates.reverse(),
        r=>{r.candidates[0].affirmative.claim.evidence="Подмена";},
        r=>{r.candidates[0].negative.citations=["[О3:Ф1:Ф8]"];},
        r=>{r.candidates[0].reviewTier="context-qualified-review";}
    ]){
        const corrupted=structuredClone(ranked);
        modify(corrupted);
        assert.equal(auditStageClaimConflictTriage(found,corrupted),null);
    }
    assert.equal(auditStageClaimConflictTriage(null,ranked),null);
});

test("benchmark refuses unlabeled, repeated and incomplete fixtures",()=>{
    assert.equal(evaluateStageClaimConflictFixtures([]),null);
    assert.equal(evaluateStageClaimConflictFixtures([
        samples[0],samples[0]]),null);
    assert.equal(evaluateStageClaimConflictFixtures([{
        ...samples[0],expectedPairs:undefined
    }]),null);
});
