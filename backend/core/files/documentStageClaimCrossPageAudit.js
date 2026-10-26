const REF=/^\[О([1-9]\d{0,2}):Ф1:Ф[1-9]\d{0,2}\]$/u;

/** Verify every cross-window contradiction citation against signed source locations. */
export function auditStageClaimCrossPageConflicts(triage,verified){
    if(!Array.isArray(triage?.candidates)||
       triage.candidates.length>12||
       triage.possibleConflictPairs!==triage.candidates.length||
       triage.allCandidatesRetained!==true||
       !Array.isArray(verified?.windows))return null;
    let crossSection=0,crossStage=0,adjacent=0,citations=0;
    const pairs=[];
    for(const pair of triage.candidates){
        const [left,right]=[pair?.affirmative,pair?.negative];
        const claims=[left,right];
        for(const item of claims){
            const c=item?.claim;
            const original=verified.windows[c?.windowNumber-1];
            if(!original||original.number!==c.windowNumber||
               original.sectionIndex!==c.sectionIndex||
               original.startOffset!==c.startOffset||
               original.endOffset!==c.endOffset||
               !Array.isArray(item.citations)||!item.citations.length||
               item.citations.some(s=>{
                   const ref=typeof s==="string"?REF.exec(s):null;
                   return !ref||Number(ref[1])!==c.windowNumber;
               }))return null;
            citations+=item.citations.length;
        }
        if(left.claim.windowNumber===right.claim.windowNumber)return null;
        const distinctSections=left.claim.sectionIndex!==right.claim.sectionIndex;
        const distinctStages=left.claim.stageNumber!==right.claim.stageNumber;
        if(distinctSections)crossSection++;
        if(distinctStages)crossStage++;
        if(Math.abs(left.claim.windowNumber-right.claim.windowNumber)===1)
            adjacent++;
        pairs.push({affirmativeWindow:left.claim.windowNumber,
            negativeWindow:right.claim.windowNumber,
            affirmativeSection:left.claim.sectionIndex,
            negativeSection:right.claim.sectionIndex,
            crossesSection:distinctSections,crossesStage:distinctStages});
    }
    return {crossPageCandidatePairs:pairs.length,
        crossSectionCandidatePairs:crossSection,
        crossStageCandidatePairs:crossStage,
        adjacentWindowCandidatePairs:adjacent,
        crossPageCandidateCitationsAudited:citations,
        crossPageCandidateAnchorsVerified:true,
        crossPageCandidateLocations:pairs,
        crossPageSemanticConflictVerified:false,
        unselectedClaimsChecked:false};
}
