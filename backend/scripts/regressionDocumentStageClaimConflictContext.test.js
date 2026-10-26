import test from "node:test";
import assert from "node:assert/strict";
import {groupStageClaimSelections}
 from "../core/files/documentStageClaimSelectionGrouping.js";
import {readStageClaimContext,compareStageClaimConflictContext}
 from "../core/files/documentStageClaimConflictContext.js";
import {findStageClaimConflictCandidates}
 from "../core/files/documentStageClaimConflictCandidates.js";
import {renderStageClaimConflictCandidates}
 from "../core/files/documentStageClaimConflictRender.js";

const entry=(text,window,quote)=>({
    stageNumber:1,windowNumber:window,sectionIndex:1,
    startOffset:100*window,endOffset:100*window+90,
    claim:"• Позиция автора: "+text+" [О"+window+":Ф1:Ф1]",
    evidence:"  Основание: «"+quote+"»"
});

test("checks distinct dates, conditions, and exceptions in signed-source wording",()=>{
    const p=entry("Договор действует",1,
        "На 01.01.2024 договор действует при наличии решения");
    const n=entry("Договор не действует",2,
        "На 01.01.2025 договор не действует при наличии согласия, за исключением спорных случаев");
    const conflicts=findStageClaimConflictCandidates(
        groupStageClaimSelections([p,n]));
    assert.equal(conflicts.possibleConflictPairs,1);
    assert.equal(conflicts.contextQualifiedPairs,1);
    assert.equal(conflicts.contextReviewPerformed,true);
    const context=conflicts.candidates[0].context;
    assert.deepEqual(context.affirmative.dates,["01.01.2024"]);
    assert.deepEqual(context.negative.dates,["01.01.2025"]);
    assert.ok(context.differences.includes("different-explicit-dates"));
    assert.ok(context.differences.includes("different-explicit-conditions"));
    assert.ok(context.differences.includes("exception-qualification-present"));
    assert.equal(context.reviewStatus,"context-qualifiers-need-review");
    assert.equal(context.semanticConflictVerified,false);
    assert.equal(context.fullSourceContextReviewed,false);
    const rendered=renderStageClaimConflictCandidates(conflicts);
    for(const item of [p,n]){
        assert.ok(rendered.includes(item.claim));
        assert.ok(rendered.includes(item.evidence));
        assert.ok(rendered.includes("смещения "+item.startOffset+"–"+item.endOffset));
    }
    assert.ok(rendered.includes("01.01.2024"));
    assert.ok(rendered.includes("01.01.2025"));
    assert.ok(rendered.includes("при наличии решения"));
    assert.ok(rendered.includes("при наличии согласия"));
    assert.ok(rendered.includes("смысловое противоречие не подтверждено"));
});

test("one-sided exception or condition remains unresolved instead of disproving conflict",()=>{
    const p=entry("Договор действует",1,"Договор действует");
    const n=entry("Договор не действует",2,
        "Договор не действует, если наступило обстоятельство, кроме отменённых случаев");
    const result=findStageClaimConflictCandidates(groupStageClaimSelections([p,n]));
    assert.equal(result.contextQualifiedPairs,1);
    assert.ok(result.candidates[0].context.differences.includes(
        "condition-mentioned-on-one-side"));
    assert.ok(result.candidates[0].context.differences.includes(
        "exception-qualification-present"));
    assert.equal(result.candidates[0].context.semanticConflictVerified,false);
});

test("no explicit qualifiers gives unknown scope rather than certified contradiction",()=>{
    const p=entry("Договор действует",1,"Текст источника");
    const n=entry("Договор не действует",2,"Другой текст источника");
    const result=findStageClaimConflictCandidates(groupStageClaimSelections([p,n]));
    assert.equal(result.possibleConflictPairs,1);
    assert.equal(result.contextQualifiedPairs,0);
    assert.equal(result.candidates[0].context.reviewStatus,
        "context-insufficient-to-resolve");
    assert.deepEqual(readStageClaimContext({claim:p}),{
        dates:[],conditions:[],exceptionMentioned:false
    });
    assert.equal(compareStageClaimConflictContext(null),null);
});

test("dates in citation coordinates cannot masquerade as cited historical dates",()=>{
    const p=entry("Договор действует",20,"Текст без дат");
    const ctx=readStageClaimContext({claim:p});
    assert.deepEqual(ctx.dates,[]);
});

test("matching explicit dates are not mislabelled as differing dates",()=>{
    const p=entry("Договор действует",1,"Действует на 01.02.2024");
    const n=entry("Договор не действует",2,"Не действует на 01.02.2024");
    const r=findStageClaimConflictCandidates(groupStageClaimSelections([p,n]));
    assert.equal(r.possibleConflictPairs,1);
    assert.deepEqual(r.candidates[0].context.differences,[]);
    assert.equal(r.candidates[0].context.fullSourceContextReviewed,false);
});
