const REF=/\[О[1-9]\d{0,2}:Ф1:Ф[1-9]\d{0,2}\]/gu;

/** Group only identical source wording; preserve each signed occurrence and citation. */
export function groupStageClaimSelections(selected){
    if(!Array.isArray(selected)||!selected.length||selected.length>768)
        return null;
    const groups=[],byText=new Map();
    for(const claim of selected){
        if(typeof claim?.claim!=="string"||
           typeof claim?.evidence!=="string"||!claim.evidence.trim()||
           !Number.isSafeInteger(claim.stageNumber)||
           !Number.isSafeInteger(claim.windowNumber)||
           !Number.isSafeInteger(claim.sectionIndex))return null;
        const citations=[...claim.claim.matchAll(REF)].map(match=>match[0]);
        const statement=claim.claim.replace(REF,"")
            .replace(/\s+/gu," ").trim();
        if(!statement||!citations.length)return null;
        let group=byText.get(statement);
        if(!group){
            group={statement,occurrences:[]};
            byText.set(statement,group);
            groups.push(group);
        }
        group.occurrences.push({claim,citations});
    }
    return {groups,selectedClaims:selected.length,
        exactTextGroups:groups.length,
        repeatedFormulations:selected.length-groups.length,
        groupingBasis:"exact-source-wording-excluding-citation-ids",
        semanticEquivalenceVerified:false};
}
