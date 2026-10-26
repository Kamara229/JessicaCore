import test from "node:test";
import assert from "node:assert/strict";
import {inspectStageClaimWindowContinuity}
 from "../core/files/documentStageClaimWindowContinuity.js";
import {renderStageClaimWindowContinuity}
 from "../core/files/documentStageClaimWindowContinuityRender.js";

const verified={complete:true,allSourceWindowsVerified:true,
    sourceChars:24,totalWindows:3,
    windows:[
        {number:1,sectionIndex:0,startOffset:0,endOffset:8},
        {number:2,sectionIndex:0,startOffset:8,endOffset:16},
        {number:3,sectionIndex:1,startOffset:16,endOffset:24}
    ]};
const pick=n=>{
    const w=verified.windows[n-1];
    return {...w,windowNumber:n,stageNumber:n===3?2:1,
        claim:"• Позиция автора: Право действует [О"+n+":Ф1:Ф1]",
        evidence:"  Основание: «фрагмент "+n+"»"};
};

test("structural continuity tracks selected page gaps and section boundaries",()=>{
    const report=inspectStageClaimWindowContinuity([pick(1),pick(3)],
        verified);
    assert.equal(report?.sourceWindowsContiguous,true);
    assert.equal(report.sectionStructureAudited,true);
    assert.equal(report.sourceSectionCount,2);
    assert.equal(report.selectedSectionCount,2);
    assert.equal(report.sectionTransitions,1);
    assert.deepEqual(report.sectionTransitionsLocated,[{
        fromWindow:2,toWindow:3,fromSection:0,toSection:1
    }]);
    assert.equal(report.windowsWithoutSelectedClaims,1);
    assert.equal(report.gapsBetweenSelectedWindows,1);
    assert.equal(report.semanticSectionCoherenceVerified,false);
    const summary=renderStageClaimWindowContinuity(report,{
        crossPageCandidateAnchorsVerified:true,
        crossPageCandidatePairs:1,crossSectionCandidatePairs:1,
        crossStageCandidatePairs:1,crossPageCandidateLocations:[]
    });
    assert.ok(summary.includes("проверено исходных окон 3"));
    assert.ok(summary.includes("смысловая связность разделов не подтверждена"));
});

test("matching windows with multiple selected citations remain ordered",()=>{
    const report=inspectStageClaimWindowContinuity([
        pick(1),pick(1),pick(2),pick(3)],verified);
    assert.equal(report.selectedClaimsLocated,4);
    assert.equal(report.selectedWindowsLocated,3);
    assert.equal(report.windowsWithoutSelectedClaims,0);
    assert.equal(report.gapsBetweenSelectedWindows,0);
});

test("reordered, forged and incomplete long document evidence fails closed",()=>{
    assert.equal(inspectStageClaimWindowContinuity(
        [pick(3),pick(1)],verified),null);
    assert.equal(inspectStageClaimWindowContinuity(
        [pick(1),{...pick(3),startOffset:17}],verified),null);
    assert.equal(inspectStageClaimWindowContinuity(
        [pick(1)],{...verified,sourceChars:25}),null);
    assert.equal(inspectStageClaimWindowContinuity(
        [pick(1)],{...verified,complete:false}),null);
    const broken=structuredClone(verified);
    broken.windows[1].startOffset=9;
    assert.equal(inspectStageClaimWindowContinuity([pick(1)],broken),null);
    assert.equal(inspectStageClaimWindowContinuity([],
        verified),null);
});
