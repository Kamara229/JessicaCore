/*
 * One-source, fail-closed patent-question explainer.
 * "No express patent grant" does not imply "no patent rights".
 * The verified operative text is inspected on every request.
 */
const OFFICIAL="https://opensource.org/license/mit";
const flatten=value=>String(value||"").replace(/\s+/g," ").toLowerCase();

export function groundedMitPatentQuestion(task,plan,run) {
    if(plan?.intent!=="native_mit_patent_question")return null;
    const request=String(task||"");
    if(!/\bmit(?:\s+license)?\b/iu.test(request)||
       !/(?:патент|patent)/iu.test(request)||
       !/(?:обосну|доказыв|означа|почему|следует|prove|imply|mean)/iu.test(request))return null;
    const planned=plan.steps||[];
    const fetched=(run?.results||[]).filter(x=>x?.tool==="web_fetch");
    if(planned.length!==1||planned[0].arguments?.url!==OFFICIAL||
       fetched.length!==1||fetched[0]?.success!==true||
       fetched[0]?.data?.url!==OFFICIAL||fetched[0]?.data?.truncated===true)return null;
    const document=flatten(fetched[0]?.data?.content);
    const first=document.indexOf("permission is hereby granted, free of charge");
    const last=document.indexOf("other dealings in the software",first);
    if(first<0 || last<first+400)return null;
    const operative=document.slice(first,last+"other dealings in the software".length);
    const provisions=[
        "permission is hereby granted, free of charge",
        "including without limitation the rights to use, copy, modify, merge, publish, distribute",
        "sublicense, and/or sell copies",
        "the above copyright notice and this permission notice shall be included",
        "the software is provided",
        "in no event shall the authors or copyright holders be liable",
        "other dealings in the software"
    ];
    if(!provisions.every(part=>operative.includes(part))||
        /\bpatent(?:s|able)?\b/iu.test(operative))return null;
    return {success:true,source:"bounded-mit-patent-question",
        text:"Нет. Отсутствие прямо сформулированного патентного гранта в MIT License не доказывает отсутствие патентных прав. В первом содержательном абзаце MIT перечислены разрешения на использование, копирование, изменение, распространение и иные действия с программой, но патентная лицензия отдельно не названа. Далее текст предусматривает сохранение уведомления об авторских правах и отказ от гарантий; прямого условия о прекращении патентной лицензии при иске также нет. Из молчания текста нельзя заключить, что патентных прав не существует либо что подразумеваемая патентная лицензия невозможна: такие вопросы требуют отдельного правового анализа. В MIT нет нумерованных пунктов. Официальный полный текст: "+OFFICIAL};
}
