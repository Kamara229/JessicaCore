/*
 * Evidence-first material for a comparison of two independently fetched
 * documents. This extracts original-language quotations only: it does
 * NOT assert an interpretation or infer that an unquoted clause is absent.
 * A later Composer still has to compare the legal meaning of the text.
 */

const MAX_QUOTE=205;
const MAX_PER_CELL=2;
const MAX_LEDGER=3450;
const CLASSES=[
    {
        id:"permissions", label:"Использование и изменение",
        asked:/использ|модифиц|изменени|копир|создан|разреш|permissions?|modif|derivative/iu,
        source:[/\b(?:reproduce|prepare derivative works|permission to use|use in source|modified forms|modify|modification)\b/giu]
    },
    {
        id:"distribution",label:"Распространение",
        asked:/распростран|передач|distribution|redistribution/iu,
        source:[/\bredistribution(?:s)?\b/giu,/\bdistribut(?:e|ion)\b/giu]
    },
    {
        id:"notices",label:"Уведомления и авторство",
        asked:/уведомлен|копирайт|авторск|copyright|notice|attribution/iu,
        source:[/\b(?:copyright notice|retain the above copyright|copy of this license|attribution notices|notice file)\b/giu]
    },
    {
        id:"patents",label:"Патентные условия",
        asked:/патент|patent/iu,
        // Favor both the positive grant and potential termination condition.
        source:[/patent litigation/giu,/patent licenses?[^.]{0,65}terminat/giu,/grant of patent license/giu,/\bpatent license\b/giu]
    },
    {
        id:"restrictions",label:"Ограничения и гарантии",
        asked:/ограничен|гаранти|ответствен|товарн|бренд|имя|restriction|trademark|warranty|liability/iu,
        source:[/\b(?:trademarks?|trade names?|endorse|promote products|as is|without warranties|limitation of liability)\b/giu]
    }
];

function sourceKey(url) {
    try {
        const x=new URL(String(url||""));
        if(!["http:","https:"].includes(x.protocol))return "";
        return x.hostname.toLowerCase()+x.pathname.replace(/\/+$/,"").toLowerCase();
    }catch{return "";}
}
function quoteNear(text,pos,length) {
    let start=Math.max(0,pos-42);
    let end=Math.min(text.length,Math.max(pos+length+105,start+MAX_QUOTE));
    if(end-start>MAX_QUOTE)end=start+MAX_QUOTE;
    if(start>0){const cut=text.slice(start,Math.min(start+42,end)).search(/\s/);if(cut>=0)start+=cut+1;}
    if(end<text.length){const endCut=text.lastIndexOf(" ",end);if(endCut>start+75)end=endCut;}
    return text.slice(start,end).replace(/\s+/gu," ").trim();
}
function pickExcerpts(text,patterns) {
    const results=[];
    const starts=[];
    const original=String(text||"");
    for(const re of patterns) {
        re.lastIndex=0;
        let match;
        let examined=0;
        while((match=re.exec(original))&&examined++<35) {
            const at=match.index;
            if(starts.some(s=>Math.abs(s-at)<100))continue;
            const quote=quoteNear(original,at,match[0].length);
            if(quote.length<28)continue;
            starts.push(at);
            results.push(quote);
            if(results.length>=MAX_PER_CELL)return results;
        }
    }
    return results;
}

export function createComparisonEvidenceLedger(task,taskRunResult) {
    const t=String(task||"");
    if(!/(?:сравн|сопостав|compar)/iu.test(t))return null;
    const sources=(taskRunResult?.results||[]).filter(r=>
        r?.tool==="web_fetch"&&r?.success===true&&r?.data?.url&&
        typeof r.data?.content==="string"&&r.data.content.trim()&&r.data?.truncated!==true
    );
    if(sources.length!==2)return null;
    const ids=sources.map(r=>sourceKey(r.data.url));
    if(!ids[0]||!ids[1]||ids[0]===ids[1])return null;
    const criteria=CLASSES.filter(c=>c.asked.test(t));
    if(criteria.length<2)return null;
    const rows=criteria.map(c=>({
        id:c.id,label:c.label,
        excerpts:sources.map(s=>pickExcerpts(s.data.content,c.source))
    }));
    if(rows.every(row=>row.excerpts.every(x=>!x.length)))return null;
    return {
        sources:sources.map(s=>({url:s.data.url,title:String(s.data.title||"").slice(0,130)})),
        rows
    };
}
export function formatComparisonEvidenceLedger(ledger) {
    if(!ledger)return "";
    const lines=[
        "КАРТОЧКИ ПЕРВОИСТОЧНИКОВ ПО ЗАПРОШЕННЫМ КРИТЕРИЯМ:",
        "Это ДОСЛОВНЫЕ ВЫДЕРЖКИ из полученных документов, а не готовые выводы.",
        "Сходство слов не доказывает сходство юридических условий.",
        "Если цитата не извлечена, нельзя заявлять об отсутствии пункта в полном документе."
    ];
    for(const [i,s] of ledger.sources.entries())lines.push("ИСТОЧНИК "+(i+1)+": "+s.url);
    for(const r of ledger.rows) {
        lines.push("Критерий: "+r.label);
        for(let i=0;i<2;i++) {
            if(r.excerpts[i].length) {
                for(const q of r.excerpts[i].slice(0,r.id==="patents"?2:1))lines.push("  "+(i+1)+": «"+q+"»");
            }else{
                lines.push("  "+(i+1)+": релевантная цитата не выделена, условие не проверено");
            }
        }
    }
    return lines.join("\n").slice(0,MAX_LEDGER);
}
export function renderComparisonEvidenceLedger(ledger) {
    if(!ledger)return null;
    const meaningful=ledger.rows.filter(r=>r.excerpts.some(x=>x.length));
    if(meaningful.length<2 || meaningful.every(r=>r.excerpts.some(x=>!x.length)))return null;
    const lines=[
        "Частичный результат сравнения",
        "",
        "Полный ответ не прошёл смысловую проверку. Ниже приведены только дословные выдержки из двух загруженных документов, а не окончательные юридические выводы.",
        "",
        "| Критерий | Источник 1: выдержки | Источник 2: выдержки |",
        "| --- | --- | --- |"
    ];
    for(const r of meaningful){
        const cells=r.excerpts.map(q=>q.length
            ?q.map(x=>"«"+x.replace(/\|/g,"¦")+"»").join("<br>")
            :"Не удалось подтвердить по выделенным выдержкам");
        lines.push("| "+r.label+" | "+cells[0]+" | "+cells[1]+" |");
    }
    lines.push("","Оригиналы:", "1. "+ledger.sources[0].url, "2. "+ledger.sources[1].url,
        "","Выдержки подтверждают наличие текста, но не все сравнения его смысла. Отсутствие цитаты не означает отсутствие условия в оригинале.");
    return {text:lines.join("\n"), rows:meaningful.length, degraded:true};
}
