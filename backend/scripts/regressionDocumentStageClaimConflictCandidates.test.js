import test from "node:test";
import assert from "node:assert/strict";
import {groupStageClaimSelections}
 from "../core/files/documentStageClaimSelectionGrouping.js";
import {findStageClaimConflictCandidates}
 from "../core/files/documentStageClaimConflictCandidates.js";
import {renderStageClaimConflictCandidates}
 from "../core/files/documentStageClaimConflictRender.js";

const entry=(body,window)=>({
    stageNumber:window>16?2:1,
    windowNumber:window,sectionIndex:1,
    startOffset:window*100,endOffset:window*100+90,
    claim:"• Позиция автора: "+body+" [О"+window+":Ф1:Ф1]",
    evidence:"  Основание: «Подлинная цитата "+window+"»"
});
const check=claims=>findStageClaimConflictCandidates(
    groupStageClaimSelections(claims));

test("flags only a possible opposite-negation pair across source windows",()=>{
    const claims=[entry("Договор действует",1),
        entry("Договор не действует",2),
        entry("Срок составляет десять дней",3)];
    const result=check(claims);
    assert.equal(result?.possibleConflictPairs,1);
    assert.equal(result.semanticConflictVerified,false);
    assert.equal(result.unselectedClaimsChecked,false);
    const text=renderStageClaimConflictCandidates(result);
    for(const claim of claims.slice(0,2)){
        assert.ok(text.includes(claim.claim));
        assert.ok(text.includes(claim.evidence));
        assert.ok(text.includes("смещения "+claim.startOffset+"–"+claim.endOffset));
    }
    assert.ok(text.includes("проверить контекст"));
    assert.ok(!text.includes("Срок составляет десять дней"));
});

test("same-window claims and contextual negation are not conflicts",()=>{
    for(const claims of [
        [entry("Договор действует",1),entry("Договор не действует",1)],
        [entry("Договор действует",1),entry("Договор не всегда действует",2)],
        [entry("Договор действует",1),entry("Договор не действует при расторжении",2)],
        [entry("Договор действует",1),entry("Договор действует",2)],
        [entry("Право не действует и не применяется",1),entry("Право действует и не применяется",2)]
    ]){
        const result=check(claims);
        assert.equal(result?.possibleConflictPairs,0);
        assert.ok(renderStageClaimConflictCandidates(result)
            .includes("не означает отсутствия других противоречий"));
    }
});

test("explicit modal negation is now recognized without asserting factual conflict",()=>{
    const r=check([entry("Договор может действовать",1),
        entry("Договор не может действовать",2)]);
    assert.equal(r.possibleConflictPairs,1);
    assert.equal(r.controlledParaphrasePairs,1);
    assert.equal(r.semanticConflictVerified,false);
});

test("limit candidates rather than implying exhaustive contradiction coverage",()=>{
    const claims=[];
    for(let i=1;i<=15;i++)claims.push(entry("Договор является действительным",i));
    for(let i=17;i<=30;i++)claims.push(entry("Договор не является действительным",i));
    const result=check(claims);
    assert.equal(result.possibleConflictPairs,12);
    assert.equal(result.candidateLimitReached,true);
    assert.ok(renderStageClaimConflictCandidates(result).includes("лимит выборки"));
});

test("reject missing grouping instead of promoting unverified model assertions",()=>{
    assert.equal(findStageClaimConflictCandidates(null),null);
    assert.equal(renderStageClaimConflictCandidates(null),null);
});
