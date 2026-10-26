import test from "node:test";
import assert from "node:assert/strict";
import {readStageClaimParaphrasePolarity}
 from "../core/files/documentStageClaimConflictParaphrases.js";
import {groupStageClaimSelections}
 from "../core/files/documentStageClaimSelectionGrouping.js";
import {findStageClaimConflictCandidates}
 from "../core/files/documentStageClaimConflictCandidates.js";
import {triageStageClaimConflictCandidates}
 from "../core/files/documentStageClaimConflictTriage.js";
import {renderStageClaimConflictCandidates}
 from "../core/files/documentStageClaimConflictRender.js";

const entry=(text,window,quote)=>({
    stageNumber:1,windowNumber:window,sectionIndex:1,
    startOffset:window*100,endOffset:window*100+90,
    claim:"• Позиция автора: "+text+" [О"+window+":Ф1:Ф1]",
    evidence:"  Основание: «"+(quote||text)+"»"
});
const detect=(a,b,quoteA,quoteB)=>{
    const grouping=groupStageClaimSelections([
        entry(a,1,quoteA),entry(b,2,quoteB)]);
    return triageStageClaimConflictCandidates(
        findStageClaimConflictCandidates(grouping));
};

test("closed legal wording rules detect four previously missed synthetic forms",()=>{
    for(const [yes,no] of [
        ["Договор вступил в силу","Договор не вступил в силу"],
        ["Исполнение разрешается","Исполнение не разрешается"],
        ["Договор может действовать","Договор не может действовать"],
        ["Порядок обязателен","Порядок необязателен"],
        ["Сторона имеет право отказаться","Сторона не вправе отказаться"],
        ["Сторона вправе отказаться","Сторона не имеет права отказаться"]
    ]){
        const yesForm=readStageClaimParaphrasePolarity(
            "• Позиция автора: "+yes);
        const noForm=readStageClaimParaphrasePolarity(
            "• Позиция автора: "+no);
        assert.equal(yesForm?.key,noForm?.key);
        assert.equal(yesForm?.negative,false);
        assert.equal(noForm?.negative,true);
        const result=detect(yes,no);
        assert.equal(result.possibleConflictPairs,1);
        assert.equal(result.controlledParaphrasePairs,1);
        assert.equal(result.candidates[0].context.semanticConflictVerified,false);
        const rendered=renderStageClaimConflictCandidates(result);
        assert.ok(rendered.includes(yes));
        assert.ok(rendered.includes(no));
        assert.ok(rendered.includes("[О1:Ф1:Ф1]"));
        assert.ok(rendered.includes("[О2:Ф1:Ф1]"));
        assert.ok(rendered.includes("смещения 100–190"));
        assert.ok(rendered.includes("смещения 200–290"));
    }
});

test("no uncontrolled synonym equivalence or modal inference",()=>{
    for(const [a,b] of [
        ["Договор действует","Договор не может действовать"],
        ["Одна сторона вправе отказаться","Другая сторона не вправе отказаться"],
        ["Договор вступил в силу","Договор не действует"],
        ["Сторона вправе отказаться","Сторона не вправе изменить"],
        ["Сторона может действовать","Сторона не обязана действовать"],
        ["Порядок обязателен","Договор необязателен"],
        ["Право не действует и не применяется",
         "Право действует и не применяется"],
        ["Договор вправе действовать","Договор не действует"]
    ]){
        assert.equal(detect(a,b).possibleConflictPairs,0,
            "False lexical inference: "+a+" / "+b);
    }
});

test("context conditions and historical dates remain unresolved, with source links",()=>{
    const result=detect("Договор вступил в силу",
        "Договор не вступил в силу",
        "Договор вступил в силу с 01.01.2024",
        "Договор не вступил в силу с 01.01.2025, за исключением случая отказа");
    assert.equal(result.possibleConflictPairs,1);
    assert.equal(result.contextDeferredPairs,1);
    assert.equal(result.allCandidatesRetained,true);
    assert.equal(result.falsePositiveVerified,false);
    const ctx=result.candidates[0].context;
    assert.ok(ctx.differences.includes("different-explicit-dates"));
    assert.ok(ctx.differences.includes("exception-qualification-present"));
    assert.equal(ctx.semanticConflictVerified,false);
});

test("excessive and malformed source statements are not normalized",()=>{
    assert.equal(readStageClaimParaphrasePolarity("Договор вступил в силу"),null);
    assert.equal(readStageClaimParaphrasePolarity("• Позиция автора: "+
        "Договор ".repeat(420)+"вступил в силу"),null);
    assert.equal(readStageClaimParaphrasePolarity(
        "• Позиция автора: Договор не может не действовать"),null);
});
