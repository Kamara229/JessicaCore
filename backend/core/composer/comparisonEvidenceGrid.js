/*
 * Source-first evidence extraction for any two retrieved text documents.
 * These are verbatim EXCERPTS, never legal conclusions or verified claims.
 * The source IDs tie each excerpt to its original URL.
 */
const TOPICS=[
    {id:"use",title:"Использование и изменения",
        ask:/использов|модифиц|изменен|производн|derivativ|permiss|\buse\b/iu,
        terms:[/\buse\b|\busing\b/iu,/reproduc|cop(y|ies)|modif|derivativ/iu,/permission|grant.*copyright/iu]},
    {id:"distribution",title:"Распространение",
        ask:/распростран|передач|дистрибуц|distribut|redistribut/iu,
        terms:[/redistribut|distribut/iu,/source\s+(?:code|form)|binary\s+(?:form|code)/iu,/sublicens|provide.*copy/iu]},
    {id:"notices",title:"Уведомления и авторство",
        ask:/уведомлен|авторств|авторск|сохранен|копирайт|notice|copyright|attribut/iu,
        terms:[/copyright|notice|attribution/iu,/retain|preserv|include.*cop(y|ies)|include.*license/iu,/redistribut|distribut/iu]},
    {id:"patents",title:"Патентные условия",
        ask:/патент|patent/iu,
        terms:[/patent/iu,/litigation|terminat|institute/iu,/grant|license/iu]},
    {id:"restrictions",title:"Ограничения",
        ask:/ограничен|бренд|имен|назван|товарн|знак|trademark|endorse|restrict/iu,
        terms:[/trademark|trade\s+names?|endorse|promot/iu,/restric|permission|may\s+not|neither/iu,/patent|litigation/iu]},
    {id:"warranty",title:"Ответственность и гарантии",
        ask:/ответственн|гаранти|warrant|liabilit|disclaim/iu,
        terms:[/warrant|liabilit|disclaim/iu,/as\s+is|damages/iu]},
    {id:"obligations",title:"Условия и обязанности",
        ask:/обязанност|требован|услови|должн|обязат|condition|require/iu,
        terms:[/condition|must|shall|require/iu,/retain|provide|include|grant/iu]}
];
const MAX_SOURCE=100000;
const WINDOW=350;
const STEP=240;
const MAX_EVIDENCE_PER_TOPIC=2;

function sourceList(run){
    const items=Array.isArray(run?.results)?run.results:[];
    const fetched=items.filter(x=>x?.tool==="web_fetch"&&x?.success===true &&
        typeof x?.data?.content==="string"&&x.data.content.trim()&&x.data.url);
    return fetched.length===2?fetched:null;
}
function fragments(text){
    const raw=String(text||"").slice(0,MAX_SOURCE);
    const out=[];
    for(let start=0;start<raw.length;start+=STEP){
        let end=Math.min(raw.length,start+WINDOW);
        if(end<raw.length){
            const endWord=raw.lastIndexOf(" ",end);
            if(endWord>start+210)end=endWord;
        }
        const excerpt=raw.slice(start,end).trim();
        if(excerpt.length>28)out.push({start,text:excerpt});
        if(end>=raw.length)break;
    }
    return out;
}
function bestEvidence(text,topic){
    const scored=fragments(text).map(item=>{
        const matches=topic.terms.map(re=>re.test(item.text));
        // Prefer a provision discussing more than one relevant aspect.
        const coverage=matches.filter(Boolean).length;
        const anchor=matches[0]?3:0;
        const score=coverage*2+anchor;
        return {...item,score,coverage};
    }).filter(x=>x.coverage>0);
    scored.sort((a,b)=>b.score-a.score||a.start-b.start);
    const picked=[];
    for(const item of scored){
        // Do not count overlapping excerpts as two independent provisions.
        if(picked.some(x=>Math.abs(x.start-item.start)<WINDOW-30))continue;
        picked.push(item);
        if(picked.length>=MAX_EVIDENCE_PER_TOPIC)break;
    }
    return picked.sort((a,b)=>a.start-b.start).map(x=>x.text);
}
export function buildComparisonEvidenceGrid(task,run){
    const sources=sourceList(run);
    if(!sources)return null;
    const requested=TOPICS.filter(t=>t.ask.test(String(task||"")));
    const chosen=requested.length?requested:TOPICS.slice(0,5);
    const criteria=chosen.map(topic=>({
        id:topic.id,
        title:topic.title,
        excerpts:sources.map(s=>bestEvidence(s.data.content,topic))
    }));
    return {sources:sources.map((s,i)=>({
        id:"S"+(i+1),url:String(s.data.url),title:String(s.data.title||"")
    })),criteria};
}

export function renderComparisonEvidenceGrid(grid){
    if(!grid)return null;
    const header=[
        "КАРТА ПЕРВОИСТОЧНИКОВ — дословные выдержки, НЕ итоговые выводы.",
        "Нет цитаты по критерию ≠ в оригинале отсутствует условие.",
        ...grid.sources.map(s=>s.id+" URL: "+s.url+" Название: "+s.title)
    ];
    for(const row of grid.criteria){
        header.push("КРИТЕРИЙ: "+row.title);
        row.excerpts.forEach((quotes,sourceIndex)=>{
            header.push(grid.sources[sourceIndex].id+": "+(quotes.length?
                quotes.map((quote,i)=>"["+grid.sources[sourceIndex].id+"-"+row.id+"-"+(i+1)+"] «"+
                    quote.replace(/\s+/g," ").slice(0,WINDOW)+"»").join("; "):
                "Не удалось найти подтверждающую выдержку по этому критерию."));
        });
    }
    header.push("Ответ строй из приведённых выдержек; если они не доказывают вывод, обозначь его как непроверенный. Не делай отрицательных выводов из отсутствия цитаты.");
    return header.join("\n");
}
