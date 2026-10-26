import test from "node:test";
import assert from "node:assert/strict";
import {inspectQuotedNumericRelation}
 from "../core/files/documentNumericRelationAudit.js";
import {makeDocumentStressSource,hashDocumentStressSource}
 from "./fixtures/documentStageClaimStressCorpus.js";
import {materializeVerifiedDocumentOverflowBatches}
 from "../core/files/documentOverflowBatchBulkMaterialization.js";
import {auditDocumentExactFacts}
 from "../core/files/documentExactFactAudit.js";

test("explicit opposite numeric relations are not rescued by identical numbers",()=>{
    const quote="Показатель не более 25%";
    for(const statement of ["Показатель более 25%","Некий показатель более 25%"]){
        assert.equal(inspectQuotedNumericRelation(statement,
            "  Основание: «"+quote+"»",quote,"25%"),"contradiction");
    }
    assert.equal(inspectQuotedNumericRelation("Показатель не более 25%",
        "  Основание: «"+quote+"»",quote,"25%"),"aligned");
    assert.equal(inspectQuotedNumericRelation("Показатель менее 25%",
        "  Основание: «Показатель не менее 25%»",
        "Показатель не менее 25%","25%"),"contradiction");
    assert.equal(inspectQuotedNumericRelation("Показатель более 25%",
        "  Основание: «Показатель не менее 25%»",
        "Показатель не менее 25%","25%"),"unresolved");
    assert.equal(inspectQuotedNumericRelation("Показатель более 25%",
        "  Основание: «Показатель более 25%»",
        "Другой показатель менее 25%","25%"),"unresolved");
});

test("real 256-window source: contradictory last-window claim is flagged",()=>{
    const original=makeDocumentStressSource();
    const baseline=materializeVerifiedDocumentOverflowBatches(
        original,hashDocumentStressSource(original));
    assert.equal(baseline?.verified?.totalWindows,256);
    const replacements=baseline.verified.windows.map(w=>{
        const marker="не более "+w.number+"%";
        const local=w.text.indexOf("а".repeat(marker.length));
        assert.ok(local>=0);
        return {offset:w.startOffset+local,marker};
    });
    let source=original;
    for(const {offset,marker} of replacements.reverse())
        source=source.slice(0,offset)+marker+
            source.slice(offset+marker.length);
    const signed=materializeVerifiedDocumentOverflowBatches(
        source,hashDocumentStressSource(source));
    assert.equal(signed?.verified?.totalWindows,256);
    const claims=signed.verified.windows.map(w=>({
        stageNumber:Math.ceil(w.number/16),
        windowNumber:w.number,sectionIndex:w.sectionIndex,
        startOffset:w.startOffset,endOffset:w.endOffset,
        claim:"• В документе указано: не более "+w.number+"% [О"+w.number+":Ф1:Ф1]",
        evidence:"  Основание: «не более "+w.number+"%»"
    }));
    const good=auditDocumentExactFacts(source,signed.verified,claims);
    assert.equal(good?.selectedNumericTokensChecked,256);
    assert.equal(good.selectedNumericRelationsAligned,256);
    assert.equal(good.selectedNumericRelationContradictions,0);
    assert.equal(good.numericRelationsFullySemanticallyVerified,false);
    claims[255].claim="• В документе указано: более 256% [О256:Ф1:Ф1]";
    const forged=auditDocumentExactFacts(source,signed.verified,claims);
    assert.equal(forged?.selectedNumericTokensAbsentFromCitedWindow,0);
    assert.equal(forged?.selectedNumericTokensAbsentFromQuotedEvidence,0);
    assert.equal(forged?.selectedNumericRelationContradictions,1);
});
