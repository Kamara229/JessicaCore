import {planAuthenticatedStageClaimPage} from "./documentStageClaimPage.js";
import {verifyStageClaimSelection} from "./documentStageClaimSelectionReceipt.js";
import {renderMergedStageClaimSelections} from "./documentStageClaimSelectionMergeRender.js";
import {groupStageClaimSelections} from "./documentStageClaimSelectionGrouping.js";
import {auditGroupedStageClaimCoverage} from "./documentStageClaimSelectionCoverageAudit.js";
import {auditRenderedStageClaimCoverage} from "./documentStageClaimSelectionRenderedAudit.js";
import {findStageClaimConflictCandidates} from "./documentStageClaimConflictCandidates.js";
import {renderStageClaimConflictCandidates} from "./documentStageClaimConflictRender.js";
import {triageStageClaimConflictCandidates} from "./documentStageClaimConflictTriage.js";

/** Rebuild every page from signed reports; never trust client-supplied claims. */
export function mergeAuthenticatedStageClaimSelections({
    source,verified,reports,task,secret,selections
}={}){
    if(!Array.isArray(selections)||!selections.length||
       selections.length>128)return null;
    let offset=0,totalClaims=null;
    const selected=[];
    for(let index=0;index<selections.length;index++){
        const page=planAuthenticatedStageClaimPage({
            source,verified,reports,task,secret,claimOffset:offset
        });
        const receipt=selections[index];
        if(!page?.claims?.length||page.claimOffset!==offset||
           !verifyStageClaimSelection(page,receipt,task,secret)||
           (totalClaims!==null&&page.totalClaims!==totalClaims))
            return null;
        totalClaims=page.totalClaims;
        for(const position of receipt.selectedPositions){
            selected.push(page.claims[position-1]);
        }
        if(page.nextClaimOffset===null){
            if(index!==selections.length-1)return null;
        }else{
            if(index===selections.length-1||
               page.nextClaimOffset<=offset)return null;
            offset=page.nextClaimOffset;
        }
    }
    const grouping=groupStageClaimSelections(selected);
    if(!grouping)return null;
    const citationAudit=auditGroupedStageClaimCoverage(
        selected,grouping,verified);
    if(!citationAudit)return null;
    const groupedText=renderMergedStageClaimSelections(grouping,totalClaims);
    if(!groupedText)return null;
    const renderedAudit=auditRenderedStageClaimCoverage(groupedText,grouping);
    if(!renderedAudit||
       renderedAudit.renderedCitationOccurrencesAudited!==
           citationAudit.selectedCitationOccurrencesAudited)return null;
    const conflicts=findStageClaimConflictCandidates(grouping);
    if(!conflicts)return null;
    const triage=triageStageClaimConflictCandidates(conflicts);
    if(!triage)return null;
    const conflictText=renderStageClaimConflictCandidates(triage);
    if(!conflictText)return null;
    const text=groupedText+conflictText;
    return {success:true,mode:"document-claim-selection-merge",
        stage:"document-stage-claim-selections-verified",text,
        readingCoverage:{
            sourceSha256:selections[0].sourceSha256,
            claimPagesVerified:selections.length,totalClaims,
            selectedClaims:selected.length,
            omittedClaims:totalClaims-selected.length,
            exactTextGroups:grouping.exactTextGroups,
            repeatedFormulations:grouping.repeatedFormulations,
            groupingBasis:grouping.groupingBasis,
            ...citationAudit,...renderedAudit,
            possibleConflictPairs:conflicts.possibleConflictPairs,
            primaryConflictReviewPairs:triage.primaryReviewPairs,
            contextDeferredConflictPairs:triage.contextDeferredPairs,
            conflictCandidatesPreserved:triage.allCandidatesRetained,
            conflictFalsePositiveVerified:false,
            conflictFilterMeaning:triage.filterMeaning,
            contextQualifiedPairs:conflicts.contextQualifiedPairs,
            conflictContextReviewPerformed:conflicts.contextReviewPerformed,
            fullSourceContextReviewed:false,
            conflictCandidateLimitReached:conflicts.candidateLimitReached,
            conflictDetectionScope:conflicts.scope,
            semanticConflictVerified:false,
            semanticEquivalenceVerified:false,
            allClaimPagesVerified:true,
            reportOriginAuthenticated:true,
            selectionReceiptsAuthenticated:true,
            sourceQuotePresenceChecked:true,
            newFactGenerationPerformed:false,
            fullDocumentSynthesisPerformed:false,
            completeSourceAnalysisConfirmed:false,
            scientificCorrectnessVerified:false
        }};
}
