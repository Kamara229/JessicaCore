import test from "node:test";
import assert from "node:assert/strict";
import {groupStageClaimSelections}
 from "../core/files/documentStageClaimSelectionGrouping.js";
import {findStageClaimConflictCandidates}
 from "../core/files/documentStageClaimConflictCandidates.js";
import {triageStageClaimConflictCandidates}
 from "../core/files/documentStageClaimConflictTriage.js";
import {renderStageClaimConflictCandidates}
 from "../core/files/documentStageClaimConflictRender.js";

const entry=(window,statement,quote)=>({
    stageNumber:1,windowNumber:window,sectionIndex:1,
    startOffset:window*100,endOffset:window*100+90,
    claim:"• Позиция автора: "+statement+
        " [О"+window+":Ф1:Ф1]",
    evidence:"  Основание: «"+quote+"»"
});
const positive=entry(1,"Договор действует","На 01.01.2024 договор действует");
const dated=entry(2,"Договор не действует","На 01.01.2025 договор не действует");
const undated=entry(3,"Договор не действует","Договор не действует");

test("contextual differences are deprioritized, not silently discarded",()=>{
    const grouped=groupStageClaimSelections([positive,dated,undated]);
    const found=findStageClaimConflictCandidates(grouped);
    assert.equal(found.possibleConflictPairs,2);
    assert.equal(found.candidates[0].negative.claim.windowNumber,2);
    const result=triageStageClaimConflictCandidates(found);
    assert.equal(result.primaryReviewPairs,1);
    assert.equal(result.contextDeferredPairs,1);
    assert.equal(result.allCandidatesRetained,true);
    assert.equal(result.falsePositiveVerified,false);
    assert.equal(result.filterMeaning,"reorder-only-no-automatic-dismissal");
    assert.deepEqual(result.candidates.map(p=>p.negative.claim.windowNumber),[3,2]);
    assert.equal(result.candidates[0].reviewTier,"primary-review");
    assert.equal(result.candidates[1].reviewTier,"context-qualified-review");
    const rendered=renderStageClaimConflictCandidates(result);
    assert.ok(rendered.includes("отложены на проверку условий: 1"));
    assert.ok(rendered.includes("пониженный приоритет"));
    assert.ok(rendered.includes("01.01.2024"));
    assert.ok(rendered.includes("01.01.2025"));
    assert.equal(rendered.split(positive.claim).length,3);
    for(const source of [positive,dated,undated]){
        assert.ok(rendered.includes(source.claim));
        assert.ok(rendered.includes(source.evidence));
        assert.ok(rendered.includes("смещения "+source.startOffset+"–"+
            source.endOffset));
    }
});

test("same date or missing condition never constitutes proven contradiction",()=>{
    const second=entry(2,"Договор не действует",
        "На 01.01.2024 договор не действует");
    const found=findStageClaimConflictCandidates(
        groupStageClaimSelections([positive,second]));
    const result=triageStageClaimConflictCandidates(found);
    assert.equal(result.primaryReviewPairs,1);
    assert.equal(result.contextDeferredPairs,0);
    assert.equal(result.semanticConflictVerified,false);
});

test("empty results and bounded capacity keep explicit uncertainty",()=>{
    const empty=findStageClaimConflictCandidates(
        groupStageClaimSelections([positive]));
    const result=triageStageClaimConflictCandidates(empty);
    assert.equal(result.possibleConflictPairs,0);
    assert.equal(result.allCandidatesRetained,true);
    assert.equal(result.falsePositiveVerified,false);
    assert.ok(renderStageClaimConflictCandidates(result)
        .includes("не означает отсутствия других противоречий"));
});

test("tampered or incomplete candidate lists are rejected",()=>{
    const found=findStageClaimConflictCandidates(
        groupStageClaimSelections([positive,dated,undated]));
    const mutations=[
        item=>{item.possibleConflictPairs++;},
        item=>{item.candidates[0].context.differences=["made-up-tag"];},
        item=>{item.candidates[1].context.semanticConflictVerified=true;},
        item=>{delete item.candidates[0].affirmative;}
    ];
    for(const mutate of mutations){
        const altered=structuredClone(found);
        mutate(altered);
        assert.equal(triageStageClaimConflictCandidates(altered),null);
    }
    assert.equal(triageStageClaimConflictCandidates(null),null);
});
