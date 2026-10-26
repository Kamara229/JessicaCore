const QUALIFIERS=new Set([
    "different-explicit-dates","different-explicit-conditions",
    "condition-mentioned-on-one-side","exception-qualification-present"
]);

/** Prioritize lexical contradiction candidates; never discard signed evidence. */
export function triageStageClaimConflictCandidates(result){
    if(!Array.isArray(result?.candidates)||
       result.candidates.length>12||
       result.possibleConflictPairs!==result.candidates.length)return null;
    const primary=[],deferred=[];
    for(const pair of result.candidates){
        const context=pair?.context;
        if(!pair.affirmative?.claim||!pair.negative?.claim||
           !Array.isArray(context?.differences)||
           context.semanticConflictVerified!==false||
           context.differences.some(value=>!QUALIFIERS.has(value)))
            return null;
        const qualified=context.differences.length>0;
        const entry={...pair,reviewTier:qualified?
            "context-qualified-review":"primary-review"};
        (qualified?deferred:primary).push(entry);
    }
    const candidates=[...primary,...deferred];
    if(candidates.length!==result.candidates.length)return null;
    return {...result,candidates,
        primaryReviewPairs:primary.length,
        contextDeferredPairs:deferred.length,
        allCandidatesRetained:true,contextTriagePerformed:true,
        filterMeaning:"reorder-only-no-automatic-dismissal",
        falsePositiveVerified:false,semanticConflictVerified:false};
}
