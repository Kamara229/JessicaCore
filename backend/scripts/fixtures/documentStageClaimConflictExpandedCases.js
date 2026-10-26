/** Manually labelled synthetic review examples; never real-world truth labels. */
const claim=(window,statement,evidence)=>({
    stageNumber:1,windowNumber:window,sectionIndex:1,
    startOffset:100*window,endOffset:100*window+90,
    claim:"• Позиция автора: "+statement+" [О"+window+":Ф1:Ф1]",
    evidence:"  Основание: «"+evidence+"»"
});

function example(id,category,positive,negative,evidenceA,evidenceB,{
    expected=false,tier="primary-review",sameWindow=false
}={}){
    return {id,category,claims:[
        claim(1,positive,evidenceA),
        claim(sameWindow?1:2,negative,evidenceB)
    ],expectedPairs:expected?[{
        affirmativeWindow:1,negativeWindow:2,reviewTier:tier
    }]:[]};
}
const baseA="Договор действует",baseB="Договор не действует";

export const expandedConflictBenchmarkCases=[
    example("direct","polarity",baseA,baseB,baseA,baseB,{expected:true}),
    example("same-date","dates",baseA,baseB,
        "На 01.02.2024 договор действует",
        "На 01.02.2024 договор не действует",{expected:true}),
    example("copula","polarity","Акт является обязательным",
        "Акт не является обязательным","Акт является обязательным",
        "Акт не является обязательным",{expected:true}),
    example("admissibility","polarity","Переход допускается",
        "Переход не допускается","Переход допускается",
        "Переход не допускается",{expected:true}),
    example("statutory","polarity","Право предусмотрено",
        "Право не предусмотрено","Право предусмотрено",
        "Право не предусмотрено",{expected:true}),
    example("duty","polarity","Сторона обязана исполнить решение",
        "Сторона не обязана исполнить решение",
        "Сторона обязана исполнить решение",
        "Сторона не обязана исполнить решение",{expected:true}),
    example("compliance","polarity","Решение соответствует требованиям",
        "Решение не соответствует требованиям",
        "Решение соответствует требованиям",
        "Решение не соответствует требованиям",{expected:true}),

    example("different-dates","dates",baseA,baseB,
        "По состоянию на 01.01.2024 договор действует",
        "По состоянию на 01.01.2025 договор не действует"),
    example("different-conditions","conditions",baseA,baseB,
        "Договор действует при наличии решения",
        "Договор не действует при наличии отказа"),
    example("exception","exceptions",baseA,baseB,
        "Договор действует","Договор не действует, за исключением особых случаев"),
    example("one-sided-condition","conditions",baseA,baseB,
        "Договор действует","Договор не действует если решение отменено"),
    example("date-on-one-side","dates",baseA,baseB,
        "На 01.01.2024 договор действует",
        "Договор не действует без указания даты"),

    example("force-paraphrase","paraphrases","Договор вступил в силу",
        "Договор не вступил в силу",
        "Договор вступил в силу","Договор не вступил в силу",
        {expected:true}),
    example("permitted-paraphrase","paraphrases","Исполнение разрешается",
        "Исполнение не разрешается",
        "Исполнение разрешается","Исполнение не разрешается",
        {expected:true}),
    example("modal-paraphrase","paraphrases","Договор может действовать",
        "Договор не может действовать",
        "Договор может действовать","Договор не может действовать",
        {expected:true}),

    example("same-polarity","control",baseA,baseA,baseA,baseA),
    example("same-window","control",baseA,baseB,baseA,baseB,
        {sameWindow:true}),
    example("morphological-paraphrase","paraphrases","Порядок обязателен",
        "Порядок необязателен","Порядок обязателен",
        "Порядок необязателен",{expected:true}),
    example("additional-negation","paraphrases",
        "Право не действует и не применяется",
        "Право действует и не применяется",
        "Право не действует и не применяется",
        "Право действует и не применяется",{expected:true}),
    example("consistent-conditions","control",
        "Договор действует при наличии акта",
        "Договор не действует без акта",
        "Договор действует при наличии акта",
        "Договор не действует без акта"),
    example("date-and-exception","dates",baseA,baseB,
        "На 01.01.2024 договор действует",
        "На 01.01.2025 договор не действует, за исключением частных случаев"),
    example("review-qualifier","exceptions",baseA,baseB,
        "Договор действует",
        "Договор не действует, за исключением отдельных обстоятельств",
        {expected:true,tier:"context-qualified-review"})
];
