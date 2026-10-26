import test from "node:test";
import assert from "node:assert/strict";
import {performance} from "node:perf_hooks";
import {planDocumentOverflowBatches}
 from "../core/files/documentOverflowReadBatches.js";
import {materializeDocumentOverflowBatch}
 from "../core/files/documentOverflowBatchMaterialization.js";
import {verifyMaterializedOverflowBatches}
 from "../core/files/documentOverflowBatchVerification.js";
import {documentPlanCacheDiagnostics}
 from "../core/files/documentOverflowPlanCache.js";
import {groupStageClaimSelections}
 from "../core/files/documentStageClaimSelectionGrouping.js";
import {auditGroupedStageClaimCoverage}
 from "../core/files/documentStageClaimSelectionCoverageAudit.js";
import {renderMergedStageClaimSelections}
 from "../core/files/documentStageClaimSelectionMergeRender.js";
import {auditRenderedStageClaimCoverage}
 from "../core/files/documentStageClaimSelectionRenderedAudit.js";
import {inspectStageClaimWindowContinuity}
 from "../core/files/documentStageClaimWindowContinuity.js";
import {findStageClaimConflictCandidates}
 from "../core/files/documentStageClaimConflictCandidates.js";
import {triageStageClaimConflictCandidates}
 from "../core/files/documentStageClaimConflictTriage.js";
import {auditStageClaimConflictTriage}
 from "../core/files/documentStageClaimConflictQualityAudit.js";
import {auditStageClaimCrossPageConflicts}
 from "../core/files/documentStageClaimCrossPageAudit.js";
import {renderStageClaimConflictCandidates}
 from "../core/files/documentStageClaimConflictRender.js";
import {makeDocumentStressSource,hashDocumentStressSource,
    claimsFromStressWindows}
 from "./fixtures/documentStageClaimStressCorpus.js";

const MAX_ELAPSED_MS=180000;
const MAX_HEAP=350*1024*1024;
const MAX_RSS=900*1024*1024;
let cached=null;

function stressCase(){
    if(cached)return cached;
    const start=performance.now();
    const cacheBefore=documentPlanCacheDiagnostics();
    const source=makeDocumentStressSource();
    const hash=hashDocumentStressSource(source);
    const plan=planDocumentOverflowBatches(source,hash);
    assert.equal(plan?.totalWindows,256);
    assert.equal(plan.batchCount,16);
    assert.equal(plan.batches.length,16);
    let peakHeap=process.memoryUsage().heapUsed;
    let peakRss=process.memoryUsage().rss;
    const batches=[];
    for(let number=1;number<=16;number++){
        const next=materializeDocumentOverflowBatch(source,hash,number);
        assert.equal(next?.windowCount,16);
        batches.push(next);
        peakHeap=Math.max(peakHeap,process.memoryUsage().heapUsed);
        peakRss=Math.max(peakRss,process.memoryUsage().rss);
    }
    const verified=verifyMaterializedOverflowBatches(source,hash,batches);
    assert.equal(verified?.complete,true);
    assert.equal(verified.totalWindows,256);
    assert.equal(verified.windows.length,256);
    peakHeap=Math.max(peakHeap,process.memoryUsage().heapUsed);
    peakRss=Math.max(peakRss,process.memoryUsage().rss);
    const elapsedMs=Math.round(performance.now()-start);
    const cacheAfter=documentPlanCacheDiagnostics();
    const verifiedCacheHits=cacheAfter.hits-cacheBefore.hits;
    assert.equal(verifiedCacheHits,17);
    const metrics={windows:256,batches:16,chars:source.length,
        elapsedMs,verifiedCacheHits,
        peakHeapMiB:Math.round(peakHeap/1048576),
        peakRssMiB:Math.round(peakRss/1048576),
        measurementScope:"plan/materialize/verify, synthetic fixture; not deployment"};
    process.stdout.write("DOCUMENT_256_WINDOW_STRESS "+JSON.stringify(metrics)+"\n");
    assert.ok(elapsedMs<MAX_ELAPSED_MS,
        "Bounded local workload exceeded 180 seconds");
    assert.ok(peakHeap<MAX_HEAP,"Heap exceeded 350 MiB");
    assert.ok(peakRss<MAX_RSS,"RSS exceeded 900 MiB");
    cached={source,hash,plan,batches,verified,metrics};
    return cached;
}

test("256 actual source windows are losslessly planned, materialized, verified",()=>{
    const {source,hash,plan,verified}=stressCase();
    assert.ok(source.length>2_000_000);
    assert.equal(plan.sourceSha256,hash);
    assert.equal(verified.collectedChars,source.length);
    assert.equal(verified.verifiedBatches,16);
    assert.equal(verified.allSourceWindowsVerified,true);
    assert.equal(verified.modelAnalysisPerformed,false);
    assert.equal(verified.windows.at(-1).endOffset,source.length);
    let cursor=0,previousSection=0,transitions=0;
    for(const window of verified.windows){
        assert.equal(window.startOffset,cursor);
        assert.ok(window.endOffset>window.startOffset);
        assert.ok(source.slice(window.startOffset,window.endOffset)===window.text);
        assert.equal(hashDocumentStressSource(window.text),window.textSha256);
        if(window.sectionIndex!==previousSection)transitions++;
        previousSection=window.sectionIndex;
        cursor=window.endOffset;
    }
    assert.equal(cursor,source.length);
    assert.equal(transitions,127);
    assert.equal(verified.windows[0].number,1);
    assert.equal(verified.windows.at(-1).number,256);
});

test("256 selected claims preserve source coordinates, links and section transitions",()=>{
    const {verified,source}=stressCase();
    const selected=claimsFromStressWindows(verified.windows);
    assert.equal(selected.length,256);
    for(const claim of selected){
        const slice=source.slice(claim.startOffset,claim.endOffset);
        assert.ok(slice.includes("ааааа"));
    }
    const grouping=groupStageClaimSelections(selected);
    assert.equal(grouping?.selectedClaims,256);
    assert.equal(grouping.exactTextGroups,3);
    const coverage=auditGroupedStageClaimCoverage(selected,grouping,verified);
    assert.equal(coverage?.selectedClaimsAudited,256);
    assert.equal(coverage.selectedCitationOccurrencesAudited,256);
    const merged=renderMergedStageClaimSelections(grouping,256);
    assert.ok(merged?.includes("[О256:Ф1:Ф1]"));
    const rendered=auditRenderedStageClaimCoverage(merged,grouping);
    assert.equal(rendered?.renderedClaimsAudited,256);
    assert.equal(rendered.renderedCitationOccurrencesAudited,256);
    const continuity=inspectStageClaimWindowContinuity(selected,verified);
    assert.equal(continuity?.sourceWindowsContiguous,true);
    assert.equal(continuity.sourceSectionCount,128);
    assert.equal(continuity.selectedSectionCount,128);
    assert.equal(continuity.selectedWindowsLocated,256);
    assert.equal(continuity.sectionTransitions,127);
    assert.equal(continuity.windowsWithoutSelectedClaims,0);
    assert.equal(continuity.gapsBetweenSelectedWindows,0);
    const found=findStageClaimConflictCandidates(grouping);
    assert.equal(found.possibleConflictPairs,1);
    const triage=triageStageClaimConflictCandidates(found);
    const audit=auditStageClaimConflictTriage(found,triage);
    assert.equal(audit?.conflictReviewCandidatesLost,0);
    assert.equal(audit.conflictCitationOccurrencesPreserved,2);
    const across=auditStageClaimCrossPageConflicts(triage,verified);
    assert.equal(across?.crossPageCandidatePairs,1);
    assert.equal(across.crossStageCandidatePairs,1);
    assert.equal(across.crossSectionCandidatePairs,1);
    assert.deepEqual(across.crossPageCandidateLocations.map(p=>
        [p.affirmativeWindow,p.negativeWindow]),[[1,256]]);
    const conflictText=renderStageClaimConflictCandidates(triage);
    assert.ok(conflictText.includes("[О1:Ф1:Ф1]"));
    assert.ok(conflictText.includes("[О256:Ф1:Ф1]"));
    assert.equal(across.crossPageSemanticConflictVerified,false);
});

test("256-window limits, truncated reads and tampered source fail closed",()=>{
    const {source,hash,batches,verified}=stressCase();
    assert.equal(verifyMaterializedOverflowBatches(source,hash,
        batches.concat(batches[0])),null);
    const incomplete=verifyMaterializedOverflowBatches(source,hash,
        batches.slice(0,-1));
    assert.equal(incomplete?.complete,false);
    assert.equal(incomplete.allSourceWindowsVerified,false);
    assert.equal(incomplete.windows,undefined);
    const damaged=[...batches];
    const last=batches.at(-1);
    damaged[15]={...last,windows:[...last.windows]};
    damaged[15].windows[15]={...last.windows[15],
        text:last.windows[15].text+"ВСТАВКА"};
    assert.equal(verifyMaterializedOverflowBatches(source,hash,damaged),null);
    const selected=claimsFromStressWindows(verified.windows);
    const wrong=selected.map(c=>({...c}));
    wrong[255].startOffset++;
    assert.equal(inspectStageClaimWindowContinuity(wrong,verified),null);
    const grouped=groupStageClaimSelections(wrong);
    assert.equal(auditGroupedStageClaimCoverage(wrong,grouped,verified),null);
    assert.equal(planDocumentOverflowBatches(source,hash.slice(1)),null);
});
