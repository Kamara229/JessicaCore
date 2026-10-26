/** Render signed, source-grounded excerpts only; no semantic synthesis. */
export function renderMergedStageClaimSelections(selected,totalClaims){
    if(!Array.isArray(selected)||!selected.length||
       !Number.isSafeInteger(totalClaims)||totalClaims<selected.length)
        return null;
    let chars=0;
    const lines=[];
    for(const claim of selected){
        if(!Number.isSafeInteger(claim?.stageNumber)||
           !Number.isSafeInteger(claim?.windowNumber)||
           !Number.isSafeInteger(claim?.sectionIndex)||
           typeof claim?.claim!=="string"||
           typeof claim?.evidence!=="string")return null;
        const line="Этап "+claim.stageNumber+", окно "+
            claim.windowNumber+", раздел "+claim.sectionIndex+
            ":\n"+claim.claim+"\n"+claim.evidence;
        chars+=line.length;
        if(chars>100000)return null;
        lines.push(line);
    }
    return "Выбранные утверждения со всех подтверждённых страниц "+
        "(без новых фактов):\n\n"+lines.join("\n\n")+
        "\n\nПроверены страницы: всего утверждений "+totalClaims+
        ", выбрано "+selected.length+
        ". Проверено наличие цитат в источнике; "+
        "истинность и полнота выводов не подтверждены.";
}
