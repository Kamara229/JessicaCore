import test from "node:test";
import assert from "node:assert/strict";
import {makeDocumentStressSource,hashDocumentStressSource}
    from "./fixtures/documentStageClaimStressCorpus.js";
import {materializeVerifiedDocumentOverflowBatches}
    from "../core/files/documentOverflowBatchBulkMaterialization.js";
import {auditDocumentExactFacts}
    from "../core/files/documentExactFactAudit.js";

// Preserve every original section and window boundary; insert a unique fact per window.
function fullFactCorpus() {
    const original=makeDocumentStressSource();
    const plan=materializeVerifiedDocumentOverflowBatches(
        original,hashDocumentStressSource(original));
    assert.equal(plan?.verified?.totalWindows,256);
    const changes=plan.verified.windows.map(window=>{
        const marker=String(window.number)+"%";
        const pos=window.text.indexOf("а".repeat(marker.length));
        assert.ok(pos>=0,"no room for fact in window "+window.number);
        return {offset:window.startOffset+pos,marker};
    });
    let source=original;
    for (const {offset,marker} of changes.reverse())
        source=source.slice(0,offset)+marker+
            source.slice(offset+marker.length);
    const data=materializeVerifiedDocumentOverflowBatches(
        source,hashDocumentStressSource(source));
    assert.equal(data?.verified?.totalWindows,256);
    return {source,verified:data.verified};
}

test("256 real partition windows retain all 256 distinct literal numeric facts",()=>{
    const {source,verified}=fullFactCorpus();
    const result=auditDocumentExactFacts(source,verified);
    assert.equal(result?.exactNumericSourceWindowsAudited,256);
    assert.equal(result.exactNumericSourceWindowsWithTokens,256);
    assert.equal(result.exactNumericSourceTokensFound,256);
    assert.equal(result.exactNumericCandidateScanLimited,false);
    assert.equal(result.factExtractionSemanticCorrectnessVerified,false);
    assert.equal(result.factExtractionCompletenessVerified,false);
});

test("numeric claim from final window is checked against its exact source",()=>{
    const {source,verified}=fullFactCorpus();
    const w=verified.windows[255];
    const makeClaim=value=>({
        windowNumber:256,sectionIndex:w.sectionIndex,
        startOffset:w.startOffset,endOffset:w.endOffset,
        claim:"• В документе указано: "+value+" [О256:Ф1:Ф1]"
    });
    const okay=auditDocumentExactFacts(source,verified,[makeClaim("256%")]);
    assert.equal(okay?.selectedNumericTokensChecked,1);
    assert.equal(okay.selectedNumericTokensAbsentFromCitedWindow,0);
    const fabricated=auditDocumentExactFacts(source,verified,[makeClaim("999₽")]);
    assert.equal(fabricated?.selectedNumericTokensAbsentFromCitedWindow,1);
    assert.equal(fabricated.literalNumericTokenPresenceVerified,false);
});

test("modified source and forged window text cannot validate",()=>{
    const {source,verified}=fullFactCorpus();
    assert.equal(auditDocumentExactFacts(source+"!",verified),null);
    const changed=structuredClone(verified);
    changed.windows[255].text="forged";
    assert.equal(auditDocumentExactFacts(source,changed),null);
});
