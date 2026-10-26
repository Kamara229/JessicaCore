/** Verify each original citation, quote and offset survives rendering in order. */
export function auditRenderedStageClaimCoverage(text,grouping){
    if(typeof text!=="string"||!text||
       !Array.isArray(grouping?.groups)||!grouping.groups.length)return null;
    let cursor=0,count=0,links=0;
    for(let index=0;index<grouping.groups.length;index++){
        const group=grouping.groups[index];
        const heading="Формулировка "+(index+1)+": "+group.statement;
        const at=text.indexOf(heading,cursor);
        if(at<cursor)return null;
        cursor=at+heading.length;
        for(const entry of group.occurrences){
            const claim=entry.claim;
            const excerpt="  Этап "+claim.stageNumber+", окно "+
                claim.windowNumber+", раздел "+claim.sectionIndex+
                "; смещения "+claim.startOffset+"–"+claim.endOffset+
                "; ссылки: "+entry.citations.join(" ")+"\n"+
                claim.evidence;
            const next=text.indexOf(excerpt,cursor);
            if(next<cursor)return null;
            cursor=next+excerpt.length;
            count++;
            links+=entry.citations.length;
        }
    }
    if(count!==grouping.selectedClaims)return null;
    return {renderedClaimsAudited:count,
        renderedCitationOccurrencesAudited:links,
        renderedSourceCoordinatesAudited:count,
        selectedClaimsRenderedCompletely:true,
        unselectedClaimsAudited:false,scientificCorrectnessVerified:false};
}
