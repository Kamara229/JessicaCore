import {parentPort,workerData} from "node:worker_threads";
import {performance} from "node:perf_hooks";
import {materializeVerifiedDocumentOverflowBatches}
 from "../../core/files/documentOverflowBatchBulkMaterialization.js";
import {groupStageClaimSelections}
 from "../../core/files/documentStageClaimSelectionGrouping.js";
import {auditGroupedStageClaimCoverage}
 from "../../core/files/documentStageClaimSelectionCoverageAudit.js";
import {inspectStageClaimWindowContinuity}
 from "../../core/files/documentStageClaimWindowContinuity.js";
import {findStageClaimConflictCandidates}
 from "../../core/files/documentStageClaimConflictCandidates.js";
import {triageStageClaimConflictCandidates}
 from "../../core/files/documentStageClaimConflictTriage.js";
import {auditStageClaimCrossPageConflicts}
 from "../../core/files/documentStageClaimCrossPageAudit.js";
import {makeConcurrentStressSource,hashDocumentStressSource,
    claimsFromStressWindows} from "./documentStageClaimStressCorpus.js";

const id=workerData?.id;
const source=makeConcurrentStressSource(id);
if(!source)throw Error("Invalid concurrent synthetic corpus id");
const sha=hashDocumentStressSource(source);
parentPort.postMessage({phase:"started",id,sha});

const began=performance.now();
const merged=materializeVerifiedDocumentOverflowBatches(source,sha);
if(!merged?.verified?.complete||merged.verified.totalWindows!==256||
   merged.batches.length!==16)throw Error("Unverified 256-window corpus");
const selected=claimsFromStressWindows(merged.verified.windows);
const grouping=groupStageClaimSelections(selected);
const audit=auditGroupedStageClaimCoverage(selected,grouping,merged.verified);
const continuity=inspectStageClaimWindowContinuity(selected,merged.verified);
const candidate=findStageClaimConflictCandidates(grouping);
const triage=triageStageClaimConflictCandidates(candidate);
const cross=auditStageClaimCrossPageConflicts(triage,merged.verified);
if(audit?.selectedClaimsAudited!==256||
   audit?.selectedCitationOccurrencesAudited!==256||
   continuity?.sourceWindowsContiguous!==true||
   continuity.sectionTransitions!==127||
   cross?.crossPageCandidatePairs!==1||
   cross.crossPageCandidateLocations[0].affirmativeWindow!==1||
   cross.crossPageCandidateLocations[0].negativeWindow!==256||
   !merged.verified.windows[0].text.includes("JESSICA"+String(id))||
   !merged.verified.windows.every(window=>
       window.text===source.slice(window.startOffset,window.endOffset)))
    throw Error("Cross-document source or citation integrity failure");

parentPort.postMessage({phase:"complete",id,sha,sourceChars:source.length,
    elapsedMs:Math.round(performance.now()-began),
    heapMiB:Math.round(process.memoryUsage().heapUsed/1048576),
    windowCount:merged.verified.totalWindows,
    citationCount:audit.selectedCitationOccurrencesAudited,
    sectionCount:continuity.sourceSectionCount,
    crossPagePairs:cross.crossPageCandidatePairs,
    originalFirstWindowSha:merged.verified.windows[0].textSha256,
    originalLastWindowSha:merged.verified.windows.at(-1).textSha256,
    semanticConflictVerified:false});
