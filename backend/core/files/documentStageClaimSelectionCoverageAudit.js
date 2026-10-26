const REF=/\[О([1-9]\d{0,2}):Ф1:Ф([1-9]\d{0,2})\]/gu;
const fields=["stageNumber","windowNumber","sectionIndex",
    "startOffset","endOffset","claim","evidence"];

function inspect(claim,verified){
    if(!claim||typeof claim.claim!=="string"||
       typeof claim.evidence!=="string"||!claim.evidence.trim()||
       !Number.isSafeInteger(claim.stageNumber)||claim.stageNumber<1||
       !Number.isSafeInteger(claim.windowNumber)||claim.windowNumber<1)
        return null;
    const window=verified?.windows?.[claim.windowNumber-1];
    if(!window||window.number!==claim.windowNumber||
       claim.sectionIndex!==window.sectionIndex||
       claim.startOffset!==window.startOffset||
       claim.endOffset!==window.endOffset||
       !Number.isSafeInteger(claim.startOffset)||
       !Number.isSafeInteger(claim.endOffset)||
       claim.startOffset<0||claim.endOffset<=claim.startOffset)return null;
    const references=[...claim.claim.matchAll(REF)];
    if(!references.length||
       references.some(ref=>Number(ref[1])!==claim.windowNumber))return null;
    return {statement:claim.claim.replace(REF,"")
        .replace(/\s+/gu," ").trim(),
        refs:references.map(ref=>ref[0])};
}

/** Audit ONLY selected claims: original sources, each citation and every coordinate. */
export function auditGroupedStageClaimCoverage(selected,grouping,verified){
    if(!Array.isArray(selected)||!selected.length||
       selected.length>768||grouping?.selectedClaims!==selected.length||
       !Array.isArray(grouping?.groups))return null;
    const expected=new Map();
    let citations=0;
    for(const claim of selected){
        const parsed=inspect(claim,verified);
        if(!parsed?.statement)return null;
        if(!expected.has(parsed.statement))expected.set(parsed.statement,[]);
        expected.get(parsed.statement).push({claim,refs:parsed.refs});
        citations+=parsed.refs.length;
    }
    if(expected.size!==grouping.groups.length||
       grouping.exactTextGroups!==expected.size||
       grouping.repeatedFormulations!==selected.length-expected.size)
        return null;
    let groupIndex=0,compared=0;
    for(const [statement,occurrences] of expected){
        const group=grouping.groups[groupIndex++];
        if(group?.statement!==statement||
           !Array.isArray(group.occurrences)||
           group.occurrences.length!==occurrences.length)return null;
        for(let i=0;i<occurrences.length;i++){
            const actual=group.occurrences[i],wanted=occurrences[i];
            if(!actual||!Array.isArray(actual.citations)||
               actual.citations.length!==wanted.refs.length||
               wanted.refs.some((ref,n)=>actual.citations[n]!==ref)||
               fields.some(key=>actual.claim?.[key]!==wanted.claim[key]))
                return null;
            compared++;
        }
    }
    return {selectedClaimsAudited:compared,
        selectedCitationOccurrencesAudited:citations,
        selectedSourceCoordinatesAudited:compared,
        selectedCitationCoverageComplete:true,
        selectedSourceCoordinatesVerified:true,
        unselectedClaimsAudited:false,
        semanticEquivalenceVerified:false,claimTruthVerified:false};
}
