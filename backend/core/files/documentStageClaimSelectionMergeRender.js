/** Render each repeated formulation once, with every original source citation. */
export function renderMergedStageClaimSelections(grouping,totalClaims){
    if(!grouping||!Array.isArray(grouping.groups)||
       !grouping.groups.length||
       !Number.isSafeInteger(totalClaims)||
       totalClaims<grouping.selectedClaims)return null;
    const lines=[];
    let chars=0,occurrences=0;
    for(const [index,group] of grouping.groups.entries()){
        if(typeof group?.statement!=="string"||!group.statement||
           !Array.isArray(group.occurrences)||!group.occurrences.length)
            return null;
        const section=["Формулировка "+(index+1)+": "+group.statement];
        for(const entry of group.occurrences){
            const claim=entry?.claim,refs=entry?.citations;
            if(!Number.isSafeInteger(claim?.stageNumber)||
               !Number.isSafeInteger(claim?.windowNumber)||
               !Number.isSafeInteger(claim?.sectionIndex)||
               !Number.isSafeInteger(claim?.startOffset)||
               !Number.isSafeInteger(claim?.endOffset)||
               claim.startOffset<0||claim.endOffset<=claim.startOffset||
               typeof claim?.evidence!=="string"||
               !Array.isArray(refs)||!refs.length)return null;
            section.push("  Этап "+claim.stageNumber+", окно "+
                claim.windowNumber+", раздел "+claim.sectionIndex+
                "; смещения "+claim.startOffset+"–"+claim.endOffset+
                "; ссылки: "+refs.join(" ")+"\n"+claim.evidence);
            occurrences++;
        }
        const body=section.join("\n");
        chars+=body.length;
        if(chars>100000)return null;
        lines.push(body);
    }
    if(occurrences!==grouping.selectedClaims)return null;
    return "Утверждения из подписанных страниц сгруппированы "+
        "по дословному совпадению формулировок без ссылок.\n\n"+
        lines.join("\n\n")+"\n\nВсего утверждений "+totalClaims+
        ", выбрано "+grouping.selectedClaims+
        ", разных формулировок "+grouping.exactTextGroups+
        ". Все исходные ссылки и цитаты сохранены. "+
        "Сходство смысла, истинность и полнота выводов не проверялись.";
}
