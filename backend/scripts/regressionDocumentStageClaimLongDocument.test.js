import test from "node:test";
import assert from "node:assert/strict";
import {makeStageClaimFixture,source,secret,task}
 from "./fixtures/documentStageClaimFixture.js";
import {planAuthenticatedStageClaimPage}
 from "../core/files/documentStageClaimPage.js";
import {signStageClaimSelection}
 from "../core/files/documentStageClaimSelectionReceipt.js";
import {mergeAuthenticatedStageClaimSelections}
 from "../core/files/documentStageClaimSelectionMerge.js";
import {hashDocumentStageReportText}
 from "../core/files/documentStageReportTextHash.js";
import {signDocumentStageReceipt}
 from "../core/files/documentStageReceiptAuth.js";

const TAG="• Позиция автора: Анализ норм";
function markReport(report,mode){
    const pos=mode==="first"?report.text.indexOf(TAG):
        report.text.lastIndexOf(TAG);
    assert.ok(pos>=0);
    report.text=report.text.slice(0,pos)+
        (mode==="first"?"• Позиция автора: Договор действует":
        "• Позиция автора: Договор не действует")+
        report.text.slice(pos+TAG.length);
    report.stageReport.reportTextSha256=hashDocumentStageReportText(
        report.text);
    report.stageReport.authTag=signDocumentStageReceipt(report,secret);
    assert.ok(report.stageReport.authTag);
}
function receiptsForBoundaryWindows(verified,reports){
    const receipts=[];
    let offset=0;
    for(let i=0;i<128;i++){
        const page=planAuthenticatedStageClaimPage({
            source,verified,reports,task,secret,claimOffset:offset
        });
        assert.ok(page?.claims?.length);
        const positions=page.claims.length===1?[1]:[1,page.claims.length];
        receipts.push(signStageClaimSelection(page,positions,task,secret));
        if(page.nextClaimOffset===null)return receipts;
        offset=page.nextClaimOffset;
    }
    throw Error("Exceeded bounded signed page count");
}

test("large signed document keeps cross-section contradiction and all references",()=>{
    const {verified,reports}=makeStageClaimFixture();
    assert.ok(verified.totalWindows>16);
    assert.ok(reports.length>1);
    assert.notEqual(verified.windows[0].sectionIndex,
        verified.windows.at(-1).sectionIndex);
    const altered=structuredClone(reports);
    markReport(altered[0],"first");
    markReport(altered.at(-1),"last");
    const receipts=receiptsForBoundaryWindows(verified,altered);
    const result=mergeAuthenticatedStageClaimSelections({
        source,verified,reports:altered,task,secret,selections:receipts
    });
    assert.equal(result?.success,true);
    assert.equal(result.readingCoverage.allClaimPagesVerified,true);
    assert.equal(result.readingCoverage.sourceWindowsContiguous,true);
    assert.equal(result.readingCoverage.sectionStructureAudited,true);
    assert.ok(result.readingCoverage.sourceSectionCount>1);
    assert.ok(result.readingCoverage.sectionTransitions>0);
    assert.equal(result.readingCoverage.crossPageCandidatePairs,1);
    assert.equal(result.readingCoverage.crossSectionCandidatePairs,1);
    assert.equal(result.readingCoverage.crossStageCandidatePairs,1);
    assert.equal(result.readingCoverage.crossPageCandidateAnchorsVerified,true);
    assert.equal(result.readingCoverage.conflictReviewCandidatesLost,0);
    assert.equal(result.readingCoverage.semanticSectionCoherenceVerified,false);
    assert.equal(result.readingCoverage.crossPageSemanticConflictVerified,false);
    assert.ok(result.text.includes("Договор действует"));
    assert.ok(result.text.includes("Договор не действует"));
    assert.ok(result.text.includes("[О1:Ф1:Ф1]"));
    assert.ok(result.text.includes("[О"+verified.totalWindows+":Ф1:Ф1]"));
    assert.ok(result.text.includes("Проверка структуры длинного документа"));
    assert.ok(result.text.includes("из разных разделов 1"));
});

test("lossless signed synthetic corpus reports no false claim of semantic coverage",()=>{
    const {verified,reports}=makeStageClaimFixture();
    const receipts=receiptsForBoundaryWindows(verified,reports);
    const output=mergeAuthenticatedStageClaimSelections({
        source,verified,reports,task,secret,selections:receipts
    });
    assert.equal(output?.success,true);
    assert.equal(output.readingCoverage.crossPageCandidatePairs,0);
    assert.equal(output.readingCoverage.selectedCitationCoverageComplete,true);
    assert.equal(output.readingCoverage.sourceWindowsContiguous,true);
    assert.equal(output.readingCoverage.semanticSectionCoherenceVerified,false);
    assert.equal(output.readingCoverage.unselectedClaimsAudited,false);
});
