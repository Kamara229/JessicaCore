/** Show only original source wording, signed-stage locations and evidence. */
export function renderStageClaimConflictCandidates(result){
    if(!result||!Array.isArray(result.candidates)||
       result.candidates.length>12)return null;
    if(!result.candidates.length)
        return "\n\nПрямых пар с противоположным «не» в выбранных "+
            "утверждениях разных окон не найдено. Это не означает "+
            "отсутствия других противоречий.";
    const lines=[];
    let length=0;
    for(const [index,pair] of result.candidates.entries()){
        const entries=[pair.affirmative,pair.negative];
        const tier=pair.reviewTier;
        if(tier!==undefined&&tier!=="primary-review"&&
           tier!=="context-qualified-review")return null;
        const context=pair.context;
        if(!context||!Array.isArray(context.differences)||
           !context.affirmative||!context.negative||
           context.semanticConflictVerified!==false)return null;
        const priority=tier==="context-qualified-review"?
            "пониженный приоритет: контекстные различия":
            "первичная проверка";
        const row=["Кандидат "+(index+1)+
            " (проверить контекст; "+priority+"):",
            "Контекстные маркеры: "+
             (context.differences.length?context.differences.join(", "):
                "условия/даты/исключения не выявлены")+
             "; смысловое противоречие не подтверждено."];
        for(const [j,entry] of entries.entries()){
            const c=entry?.claim;
            if(!Number.isSafeInteger(c?.stageNumber)||
               !Number.isSafeInteger(c?.windowNumber)||
               !Number.isSafeInteger(c?.sectionIndex)||
               !Number.isSafeInteger(c?.startOffset)||
               !Number.isSafeInteger(c?.endOffset)||
               typeof c?.claim!=="string"||
               typeof c?.evidence!=="string"||
               !Array.isArray(entry?.citations)||!entry.citations.length)
                return null;
            const ctx=j===0?context.affirmative:context.negative;
            if(!Array.isArray(ctx?.dates)||
               !Array.isArray(ctx?.conditions)||
               typeof ctx.exceptionMentioned!=="boolean")return null;
            row.push((j===0?"Без отрицания":"С отрицанием")+
                ": этап "+c.stageNumber+", окно "+c.windowNumber+
                ", раздел "+c.sectionIndex+", смещения "+
                c.startOffset+"–"+c.endOffset+"\n"+
                c.claim+"\n"+c.evidence+"\n"+
                "  Контекст из цитаты: даты ["+ctx.dates.join(", ")+
                "], условия ["+ctx.conditions.join("; ")+
                "], исключение: "+(ctx.exceptionMentioned?"упомянуто":"не найдено"));
        }
        const s=row.join("\n");
        length+=s.length;
        if(length>30000)return null;
        lines.push(s);
    }
    return "\n\nВозможные текстовые противоречия "+
        "(только выбранные утверждения):\n"+lines.join("\n\n")+
        "\nНайдено пар: "+result.candidates.length+
        (result.candidateLimitReached?" (лимит выборки достигнут).":".")+
        (result.contextTriagePerformed?
            " В первичной проверке: "+result.primaryReviewPairs+
            "; отложены на проверку условий: "+result.contextDeferredPairs+
            ". Все кандидаты и ссылки сохранены.":"")+
        " Противоречие по смыслу и достоверность не подтверждены.";
}
