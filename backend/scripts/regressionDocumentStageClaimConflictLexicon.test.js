import test from "node:test";
import assert from "node:assert/strict";
import {LEGAL_CLAIM_FORMS} from "../core/files/documentStageClaimConflictLexicon.js";
import {allowLegalClaimPolarityScope}
 from "../core/files/documentStageClaimConflictScopeGuard.js";
import {readStageClaimParaphrasePolarity}
 from "../core/files/documentStageClaimConflictParaphrases.js";
import {groupStageClaimSelections}
 from "../core/files/documentStageClaimSelectionGrouping.js";
import {findStageClaimConflictCandidates}
 from "../core/files/documentStageClaimConflictCandidates.js";
import {triageStageClaimConflictCandidates}
 from "../core/files/documentStageClaimConflictTriage.js";
import {auditStageClaimConflictTriage}
 from "../core/files/documentStageClaimConflictQualityAudit.js";

const sample=(statement,window)=>({
    stageNumber:1,windowNumber:window,sectionIndex:1,
    startOffset:window*100,endOffset:window*100+80,
    claim:"• Позиция автора: "+statement+
        " [О"+window+":Ф1:Ф1]",
    evidence:"  Основание: «Установленное свидетельство "+window+"»"
});
const check=(yes,no)=> {
    const selected=[sample(yes,1),sample(no,2)];
    const grouping=groupStageClaimSelections(selected);
    const found=findStageClaimConflictCandidates(grouping);
    const triage=triageStageClaimConflictCandidates(found);
    return {selected,found,triage};
};

test("closed lexical variants preserve sources and explicit negation",()=>{
    assert.ok(LEGAL_CLAIM_FORMS.length>=7);
    for(const [yes,no] of [
        ["Исполнение разрешается","Исполнение не допускается"],
        ["Исполнение допускается","Исполнение не разрешается"],
        ["Согласие требуется","Согласие не требуется"],
        ["Применение допустимо","Применение недопустимо"],
        ["Сторона обладает правом на отказ","Сторона не вправе на отказ"],
        ["Сторона имеет право на отказ","Сторона не обладает правом на отказ"]
    ]){
        const {selected,found,triage}=check(yes,no);
        assert.equal(found.possibleConflictPairs,1);
        assert.equal(found.controlledParaphrasePairs,1);
        assert.equal(triage.allCandidatesRetained,true);
        const evidence=auditStageClaimConflictTriage(found,triage);
        assert.equal(evidence.conflictCitationOccurrencesPreserved,2);
        assert.equal(evidence.conflictReviewCandidatesLost,0);
        assert.equal(evidence.semanticConflictVerified,false);
        assert.equal(triage.candidates[0].affirmative.claim.claim,
            selected[0].claim);
        assert.equal(triage.candidates[0].negative.claim.evidence,
            selected[1].evidence);
    }
});

test("same subject, modality and clause boundaries block cross-scope matches",()=>{
    for(const [yes,no] of [
        ["Одна сторона вправе отказаться","Другая сторона не вправе отказаться"],
        ["Согласие требуется","Согласие не может требоваться"],
        ["Сторона вправе отказаться","Сторона не может отказаться"],
        ["Сторона может получить право на отказ","Сторона не имеет право на отказ"],
        ["Акт допустим","Акт не допустима"],
        ["Акт является обязательным","Акт не является обязательным при наличии запрета"]
    ]){
        const {found}=check(yes,no);
        assert.equal(found.possibleConflictPairs,0,yes+" / "+no);
    }
});

test("shared second negative clause is not dropped or equated across suffixes",()=>{
    const {found}=check("Право не действует и не применяется",
        "Право действует и не применяется");
    assert.equal(found.possibleConflictPairs,1);
    assert.equal(found.candidates[0].reason,
        "same-wording-opposite-explicit-negation");
    assert.equal(found.semanticConflictVerified,false);
    for(const pair of [
        ["Право не действует и не применяется","Право действует и применяется"],
        ["Право не действует и не возникает","Право действует и не применяется"],
        ["Право не действует и не применяется","Другое право действует и не применяется"]
    ])assert.equal(check(...pair).found.possibleConflictPairs,0);
});

test("negative and modal scope parser rejects malformed and extra modal assertions",()=>{
    assert.equal(allowLegalClaimPolarityScope({before:"Право ",
        after:" и не применяется",rule:"literal:действует",negative:true}),
    true);
    assert.equal(allowLegalClaimPolarityScope({before:"Право ",
        after:" и не возникает",rule:"literal:действует",negative:true}),
    false);
    assert.equal(allowLegalClaimPolarityScope({before:"Право ",
        after:" если может наступить событие",rule:"consent-required",
        negative:true}),false);
    assert.equal(readStageClaimParaphrasePolarity(
        "• Позиция автора: Может потребоваться согласие не требуется"),null);
    assert.equal(readStageClaimParaphrasePolarity("Согласие не требуется"),null);
});
