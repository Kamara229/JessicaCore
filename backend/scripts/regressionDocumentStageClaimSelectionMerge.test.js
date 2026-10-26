import test from "node:test";
import assert from "node:assert/strict";
import {planAuthenticatedStageClaimPage}
 from "../core/files/documentStageClaimPage.js";
import {signStageClaimSelection}
 from "../core/files/documentStageClaimSelectionReceipt.js";
import {mergeAuthenticatedStageClaimSelections}
 from "../core/files/documentStageClaimSelectionMerge.js";
import {analyzeDocumentAttachments}
 from "../core/files/documentAnalysisFacade.js";
import {materializeDocumentOverflowBatch}
 from "../core/files/documentOverflowBatchMaterialization.js";
import {pageDocumentText}
 from "../core/files/documentTextWindow.js";
import {makeStageClaimFixture,task,secret,source}
 from "./fixtures/documentStageClaimFixture.js";
import {hashDocumentStageReportText}
 from "../core/files/documentStageReportTextHash.js";
import {signDocumentStageReceipt}
 from "../core/files/documentStageReceiptAuth.js";

function buildSelections(verified,reports){
    const receipts=[],selected=[];
    let offset=0;
    for(let i=0;i<128;i++){
        const page=planAuthenticatedStageClaimPage({
            source,verified,reports,task,secret,claimOffset:offset
        });
        assert.ok(page?.claims?.length);
        const positions=page.claims.length>1?[1,2]:[1];
        receipts.push(signStageClaimSelection(page,positions,task,secret));
        selected.push(...positions.map(n=>page.claims[n-1]));
        if(page.nextClaimOffset===null)return {receipts,selected};
        offset=page.nextClaimOffset;
    }
    throw Error("Page count exceeded test bound");
}
const merge=(verified,reports,receipts,overrides={})=>
    mergeAuthenticatedStageClaimSelections({
        source,verified,reports,task,secret,selections:receipts,...overrides
    });

test("complete signed pages merge in original page order without new assertions",()=>{
    const {verified,reports}=makeStageClaimFixture();
    const {receipts,selected}=buildSelections(verified,reports);
    assert.ok(receipts.length>1);
    const result=merge(verified,reports,receipts);
    assert.equal(result.success,true);
    assert.equal(result.mode,"document-claim-selection-merge");
    assert.equal(result.readingCoverage.allClaimPagesVerified,true);
    assert.equal(result.readingCoverage.claimPagesVerified,receipts.length);
    assert.equal(result.readingCoverage.totalClaims,verified.totalWindows);
    assert.equal(result.readingCoverage.selectedClaims,selected.length);
    assert.equal(result.readingCoverage.newFactGenerationPerformed,false);
    assert.equal(result.readingCoverage.fullDocumentSynthesisPerformed,false);
    assert.equal(result.readingCoverage.completeSourceAnalysisConfirmed,false);
    assert.equal(result.readingCoverage.exactTextGroups,1);
    assert.equal(result.readingCoverage.repeatedFormulations,selected.length-1);
    assert.equal(result.readingCoverage.semanticEquivalenceVerified,false);
    assert.equal(result.readingCoverage.selectedCitationCoverageComplete,true);
    assert.equal(result.readingCoverage.selectedSourceCoordinatesVerified,true);
    assert.equal(result.readingCoverage.selectedClaimsRenderedCompletely,true);
    assert.equal(result.readingCoverage.selectedClaimsAudited,selected.length);
    assert.equal(result.readingCoverage.renderedSourceCoordinatesAudited,
        selected.length);
    assert.equal(result.readingCoverage.selectedCitationOccurrencesAudited,
        result.readingCoverage.renderedCitationOccurrencesAudited);
    assert.equal(result.readingCoverage.unselectedClaimsAudited,false);
    assert.equal(result.readingCoverage.possibleConflictPairs,0);
    assert.equal(result.readingCoverage.semanticConflictVerified,false);
    assert.equal(result.readingCoverage.conflictCandidatesPreserved,true);
    assert.equal(result.readingCoverage.primaryConflictReviewPairs,0);
    assert.equal(result.readingCoverage.contextDeferredConflictPairs,0);
    assert.equal(result.readingCoverage.conflictFalsePositiveVerified,false);
    assert.equal(result.text.split("Формулировка 1:").length,2);
    let start=0;
    for(const claim of selected){
        const marker="Этап "+claim.stageNumber+", окно "+
            claim.windowNumber+", раздел "+claim.sectionIndex+";";
        const next=result.text.indexOf(marker,start);
        assert.ok(next>=start);
        const citation=claim.claim.match(/\[О\d+:Ф1:Ф\d+\]/u)?.[0];
        assert.ok(citation);
        assert.ok(result.text.includes(citation));
        assert.ok(result.text.includes(claim.evidence));
        start=next+marker.length;
    }
});

test("missing, repeated, reordered, forged or cross-task receipts fail closed",()=>{
    const {verified,reports}=makeStageClaimFixture();
    const {receipts}=buildSelections(verified,reports);
    const tampered=structuredClone(receipts);
    tampered[0].selectedPositions=[2];
    const altered=structuredClone(receipts);
    altered[0].authTag="v1:"+"0".repeat(64);
    for(const r of [
        receipts.slice(0,-1),
        [...receipts,receipts.at(-1)],
        [receipts[1],receipts[0],...receipts.slice(2)],
        [receipts[0],receipts[0],...receipts.slice(2)],
        tampered,altered,[]
    ])assert.equal(merge(verified,reports,r),null);
    assert.equal(merge(verified,reports,receipts,{task:"Другая задача"}),null);
    assert.equal(merge(verified,reports,receipts,{secret:"b".repeat(64)}),null);
    const forgedReports=structuredClone(reports);
    forgedReports[0].text+=" выдуманный вывод";
    assert.equal(merge(verified,forgedReports,receipts),null);
});

test("overflow facade merges receipts without model calls or incomplete claim",async()=>{
    const {verified,reports}=makeStageClaimFixture();
    const {receipts}=buildSelections(verified,reports);
    const hash=reports[0].stageReport.sourceSha256;
    const first=materializeDocumentOverflowBatch(source,hash,1);
    const batches=Array.from({length:first.batchCount},(_,i)=>
        materializeDocumentOverflowBatch(source,hash,i+1));
    const pages=[];
    let offset=0;
    do{
        const page=pageDocumentText(source,offset);
        pages.push({...page,name:"paper.docx"});
        offset=page.nextOffset;
    }while(offset!==null);
    const old=process.env.JESSICA_STAGE_ATTESTATION_KEY;
    process.env.JESSICA_STAGE_ATTESTATION_KEY=secret;
    let calls=0;
    const run=receipts=>analyzeDocumentAttachments({
        task,attachments:[{name:"paper.docx",pages,
            overflowBatches:batches,stageReports:reports,
            claimSelections:receipts}]
    },async()=>{calls++;return "{}";});
    try{
        const result=await run(receipts);
        assert.equal(result.success,true);
        assert.equal(result.readingCoverage.allClaimPagesVerified,true);
        const partial=await run(receipts.slice(0,-1));
        assert.equal(partial.success,false);
        assert.equal(partial.stage,"document-stage-claim-merge-invalid");
        assert.equal(calls,0);
    }finally{
        if(old===undefined)delete process.env.JESSICA_STAGE_ATTESTATION_KEY;
        else process.env.JESSICA_STAGE_ATTESTATION_KEY=old;
    }
});

test("signed contradictory wordings reach candidate section without truth claims",()=>{
    const {verified,reports}=makeStageClaimFixture();
    const updated=structuredClone(reports);
    let seen=0;
    updated[0].text=updated[0].text.replace(
        /• Позиция автора: Анализ норм/gu,full=>{
            seen++;
            if(seen===1)return "• Позиция автора: Договор действует";
            if(seen===2)return "• Позиция автора: Договор не действует";
            return full;
        });
    assert.ok(seen>=2);
    updated[0].stageReport.reportTextSha256=
        hashDocumentStageReportText(updated[0].text);
    updated[0].stageReport.authTag=signDocumentStageReceipt(updated[0],secret);
    assert.ok(updated[0].stageReport.authTag);
    const {receipts}=buildSelections(verified,updated);
    const result=merge(verified,updated,receipts);
    assert.equal(result?.success,true);
    assert.equal(result.readingCoverage.possibleConflictPairs,1);
    assert.equal(result.readingCoverage.semanticConflictVerified,false);
    assert.equal(result.readingCoverage.selectedCitationCoverageComplete,true);
    assert.ok(result.text.includes("Возможные текстовые противоречия"));
    assert.ok(result.text.includes("Договор действует"));
    assert.ok(result.text.includes("Договор не действует"));
    assert.ok(result.text.includes("[О1:Ф1:Ф1]"));
    assert.ok(result.text.includes("[О2:Ф1:Ф1]"));
});
