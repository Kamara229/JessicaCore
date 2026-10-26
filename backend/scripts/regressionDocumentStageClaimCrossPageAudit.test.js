import test from "node:test";
import assert from "node:assert/strict";
import {auditStageClaimCrossPageConflicts}
 from "../core/files/documentStageClaimCrossPageAudit.js";

const verified={windows:[
    {number:1,sectionIndex:0,startOffset:0,endOffset:100},
    {number:2,sectionIndex:0,startOffset:100,endOffset:200},
    {number:3,sectionIndex:1,startOffset:200,endOffset:250}
]};
function claim(n,stage){
    const w=verified.windows[n-1];
    return {claim:{stageNumber:stage,windowNumber:n,
        sectionIndex:w.sectionIndex,startOffset:w.startOffset,
        endOffset:w.endOffset,claim:"• Позиция автора: Тезис [О"+
        n+":Ф1:Ф1]",evidence:"  Основание: «Цитата»"},
        citations:["[О"+n+":Ф1:Ф1]"]};
}
const ranked={allCandidatesRetained:true,possibleConflictPairs:2,
    candidates:[
        {affirmative:claim(1,1),negative:claim(3,2)},
        {affirmative:claim(1,1),negative:claim(2,1)}
    ]};

test("cross-section and cross-stage candidates retain both signed citations",()=>{
    const audit=auditStageClaimCrossPageConflicts(ranked,verified);
    assert.equal(audit?.crossPageCandidatePairs,2);
    assert.equal(audit.crossSectionCandidatePairs,1);
    assert.equal(audit.crossStageCandidatePairs,1);
    assert.equal(audit.adjacentWindowCandidatePairs,1);
    assert.equal(audit.crossPageCandidateCitationsAudited,4);
    assert.equal(audit.crossPageCandidateAnchorsVerified,true);
    assert.equal(audit.crossPageSemanticConflictVerified,false);
    assert.equal(audit.unselectedClaimsChecked,false);
    assert.deepEqual(audit.crossPageCandidateLocations[0],{
        affirmativeWindow:1,negativeWindow:3,
        affirmativeSection:0,negativeSection:1,
        crossesSection:true,crossesStage:true
    });
});

test("reject forged page numbers, citations or source section indexes",()=>{
    for(const mutation of [
        x=>{x.candidates[0].negative.claim.windowNumber=2;},
        x=>{x.candidates[0].negative.citations=["[О2:Ф1:Ф1]"];},
        x=>{x.candidates[0].negative.claim.sectionIndex=0;},
        x=>{x.candidates[1].negative.claim.endOffset=201;},
        x=>{x.candidates[1].negative.claim.windowNumber=1;},
        x=>{x.possibleConflictPairs=3;},
        x=>{x.allCandidatesRetained=false;}
    ]){
        const fake=structuredClone(ranked);
        mutation(fake);
        assert.equal(auditStageClaimCrossPageConflicts(fake,verified),null);
    }
    assert.equal(auditStageClaimCrossPageConflicts(null,verified),null);
});

test("empty cross-page set is explicitly audited rather than treated as proof",()=>{
    const result=auditStageClaimCrossPageConflicts({
        candidates:[],possibleConflictPairs:0,allCandidatesRetained:true
    },verified);
    assert.equal(result.crossPageCandidatePairs,0);
    assert.equal(result.crossPageSemanticConflictVerified,false);
    assert.equal(result.crossPageCandidateAnchorsVerified,true);
});
