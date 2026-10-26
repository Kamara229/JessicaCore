/** Render original cited claims only; never print model-generated assertions. */
export function renderStageClaimSelection(selection,page){
    if(!selection||!Array.isArray(selection.selected)||
       !selection.selected.length||
       selection.totalPageClaims!==page?.claims?.length)return null;
    const lines=[];
    for(const claim of selection.selected){
        if(!page.claims.includes(claim)||typeof claim.claim!=="string"||
           typeof claim.evidence!=="string")return null;
        lines.push("Этап "+claim.stageNumber+", окно "+claim.windowNumber+
            ", раздел "+claim.sectionIndex+":\n"+claim.claim+
            "\n"+claim.evidence);
    }
    return "Выбранные утверждения из подтверждённых этапов "+
        "(без генерации новых фактов):\n\n"+lines.join("\n\n")+
        "\n\nВыбор неполон: в странице "+page.claims.length+
        " утверждений, выбрано "+lines.length+
        ". Цитаты присутствуют в исходном тексте; истинность выводов не проверялась.";
}
