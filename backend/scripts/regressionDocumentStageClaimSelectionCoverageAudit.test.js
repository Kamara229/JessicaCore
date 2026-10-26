import test from "node:test";
import assert from "node:assert/strict";
import {groupStageClaimSelections}
 from "../core/files/documentStageClaimSelectionGrouping.js";
import {auditGroupedStageClaimCoverage}
 from "../core/files/documentStageClaimSelectionCoverageAudit.js";
import {auditRenderedStageClaimCoverage}
 from "../core/files/documentStageClaimSelectionRenderedAudit.js";
import {renderMergedStageClaimSelections}
 from "../core/files/documentStageClaimSelectionMergeRender.js";

const windows=Array.from({length:3},(_,i)=>({
    number:i+1,sectionIndex:1,startOffset:i*100,endOffset:(i+1)*100
}));
const verified={windows};
const create=(n,body,refs)=>({
    stageNumber:1,windowNumber:n,sectionIndex:1,
    startOffset:(n-1)*100,endOffset:n*100,
    claim:"• Позиция автора: "+body+" "+refs.map(f=>
        "[О"+n+":Ф1:Ф"+f+"]").join(" "),
    evidence:"  Основание: «Подтверждённый фрагмент "+n+"»"
});
const selected=[
    create(1,"Тезис", [1,2]),
    create(2,"Тезис",[1]),
    create(3,"Иной тезис",[1])
];

test("audit verifies every selected citation, offset and grouped quotation",()=>{
    const grouping=groupStageClaimSelections(selected);
    const checked=auditGroupedStageClaimCoverage(selected,grouping,verified);
    assert.equal(checked?.selectedClaimsAudited,3);
    assert.equal(checked?.selectedCitationOccurrencesAudited,4);
    assert.equal(checked?.selectedSourceCoordinatesAudited,3);
    assert.equal(checked?.selectedCitationCoverageComplete,true);
    assert.equal(checked?.unselectedClaimsAudited,false);
    const rendered=renderMergedStageClaimSelections(grouping,5);
    assert.ok(rendered.includes("смещения 0–100"));
    assert.ok(rendered.includes("смещения 200–300"));
    const output=auditRenderedStageClaimCoverage(rendered,grouping);
    assert.equal(output?.selectedClaimsRenderedCompletely,true);
    assert.equal(output?.renderedCitationOccurrencesAudited,4);
    assert.equal(output?.renderedSourceCoordinatesAudited,3);
});

test("audit rejects lost, forged, duplicate, shuffled references and coordinates",()=>{
    const mutations=[
        g=>{g.groups[0].occurrences.pop();},
        g=>{g.groups[0].occurrences.push(g.groups[0].occurrences[0]);},
        g=>{g.groups.reverse();},
        g=>{g.groups[0].occurrences[0].citations.pop();},
        g=>{g.groups[0].occurrences[0].citations[0]="[О3:Ф1:Ф1]";},
        g=>{g.groups[0].occurrences[0].claim.startOffset++;},
        g=>{g.groups[0].occurrences[0].claim.sectionIndex=9;},
        g=>{g.groups[0].occurrences[0].claim.evidence="Подмена цитаты";},
        g=>{g.groups[0].statement="Придуманный вывод";},
        g=>{g.repeatedFormulations=99;}
    ];
    for(const change of mutations){
        const grouping=structuredClone(groupStageClaimSelections(selected));
        change(grouping);
        assert.equal(auditGroupedStageClaimCoverage(
            selected,grouping,verified),null);
    }
    const mismatch=structuredClone(verified);
    mismatch.windows[0].startOffset=9;
    assert.equal(auditGroupedStageClaimCoverage(
        selected,groupStageClaimSelections(selected),mismatch),null);
});

test("rendered audit rejects missing citation, quote or source coordinate",()=>{
    const grouping=groupStageClaimSelections(selected);
    const text=renderMergedStageClaimSelections(grouping,3);
    for(const changed of [
        text.replace("[О1:Ф1:Ф2]",""),
        text.replace("смещения 100–200","смещения 100–201"),
        text.replace("«Подтверждённый фрагмент 3»","«Выдумано»"),
        text.replace("  Этап 1, окно 2","  Этап 8, окно 2"),
        text.replace("Формулировка 2:","Формулировка 9:")
    ]){
        assert.notEqual(changed,text);
        assert.equal(auditRenderedStageClaimCoverage(changed,grouping),null);
    }
});
