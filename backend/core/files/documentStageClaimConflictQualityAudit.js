function signature(pair){
    const a=pair?.affirmative,n=pair?.negative;
    if(typeof a?.claim?.claim!=="string"||
       typeof n?.claim?.claim!=="string"||
       typeof a?.claim?.evidence!=="string"||
       typeof n?.claim?.evidence!=="string"||
       !Array.isArray(a.citations)||!a.citations.length||
       !Array.isArray(n.citations)||!n.citations.length||
       !Array.isArray(pair?.context?.differences))return null;
    return JSON.stringify([a.claim,a.citations,n.claim,n.citations,
        pair.context]);
}

/** Audit lossless triage; this cannot measure real-world contradiction accuracy. */
export function auditStageClaimConflictTriage(original,triaged){
    if(!Array.isArray(original?.candidates)||
       !Array.isArray(triaged?.candidates)||
       original.candidates.length>12||
       triaged.candidates.length!==original.candidates.length||
       triaged.allCandidatesRetained!==true||
       triaged.semanticConflictVerified!==false)return null;
    const remaining=new Map();
    for(const pair of original.candidates){
        const key=signature(pair);
        if(!key)return null;
        remaining.set(key,(remaining.get(key)??0)+1);
    }
    let primary=0,deferred=0,refs=0,deferredStarted=false;
    for(const pair of triaged.candidates){
        const key=signature(pair);
        if(!key||!remaining.get(key))return null;
        remaining.set(key,remaining.get(key)-1);
        const qualified=pair.context.differences.length>0;
        if(pair.reviewTier!==
           (qualified?"context-qualified-review":"primary-review"))
            return null;
        if(qualified){deferred++;deferredStarted=true;}
        else{
            if(deferredStarted)return null;
            primary++;
        }
        refs+=pair.affirmative.citations.length+
            pair.negative.citations.length;
    }
    if([...remaining.values()].some(count=>count!==0)||
       primary!==triaged.primaryReviewPairs||
       deferred!==triaged.contextDeferredPairs)return null;
    return {conflictPairsAudited:triaged.candidates.length,
        conflictCitationOccurrencesPreserved:refs,
        conflictReviewCandidatesLost:0,
        conflictTriageRetentionVerified:true,
        realWorldFalsePositiveRateMeasured:false,
        realWorldMissedConflictsMeasured:false,
        semanticConflictVerified:false};
}
