import { extractNoticeDutyExcerpts } from "../validator/claim/noticeDutyExcerpts.js";
import { extractPatentClauseExcerpts } from "../validator/claim/patentClauseExcerpts.js";
import {extractLegalActionExcerpts} from "./sourceLegalActionExcerpts.js";
import { selectOperativeSourceText } from "../validator/claim/operativeSourceScope.js";

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
        terms:[/patent/iu,/litigation|terminat|institute/iu,/grant|license/iu],
        gate:/patent/iu},
    {id:"restrictions",title:"Ограничения",
        ask:/ограничен|бренд|имен|назван|товарн|знак|trademark|endorse|restrict/iu,
        terms:[/trademark|trade\s+names?|endorse|promot/iu,/restric|permission|may\s+not|neither/iu],
        gate:/trademark|trade\s+names?|endorse|promot|restric|may\s+not|neither/iu},
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
        typeof x?.data?.content==="string"&&x.data.content.trim()&&x.data.url&&x.data.truncated!==true);
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
    // Primary-source actions are more relevant than a bare heading
    // or a note about which translated text is authoritative.
    const actions=extractLegalActionExcerpts(text,topic.id);
    if(actions.length)return actions;

    if(topic.id==="patents") {
        const provisions=extractPatentClauseExcerpts(text,"патент");
        if(provisions.length)return provisions.slice(0,2);
    }

    // Evidence of obligations must come from operative source clauses,
    // not a neighboring copyright template or documentation appendix.
    if(topic.id==="notices") {
        const provisions=extractNoticeDutyExcerpts(text,"уведомления");
        if(provisions.length)return provisions.slice(0,2);
    }
    const scored=fragments(text).filter(item=>!topic.gate||topic.gate.test(item.text)).map(item=>{
        const matches=topic.terms.map(re=>re.test(item.text));
        // Prefer a provision discussing more than one relevant aspect.
        const coverage=matches.filter(Boolean).length;
        const anchor=matches[0]?3:0;
        const conditional=topic.id==='notices' && /if.{0,140}notice.{0,80}file/is.test(item.text) ? 12 : 0;
        const explicitDuty=topic.id==="notices" &&
            /(?:must|shall)\s+(?:retain|reproduce|give|include|provide)\b/iu.test(item.text)
            && /copyright|notice|attribut|license/iu.test(item.text) ? 24 : 0;
        // Appendix templates and how-to explanations do not establish
        // obligations in the operative document.
        const appendixTemplate=topic.id==="notices" &&
            /copyright\s*\[yyyy\]|how to apply.*license/iu.test(item.text)
            && !/(?:must|shall)\s+(?:retain|include|give)\b/iu.test(item.text) ? 24 : 0;
        const score=coverage*2+anchor+conditional+explicitDuty-appendixTemplate;
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
    // "Условия использования" is not a request for an additional
    // generic "Обязанности" row duplicating the entire comparison.
    const explicitObligations=/обязанност|обязат|требован|\brequirements?\b|\bobligations?\b/iu.test(String(task||""));
    const selected=requested.filter(t=>t.id!=="obligations"||explicitObligations);
    const chosen=selected.length?selected:requested.length?requested:TOPICS.slice(0,5);
    const criteria=chosen.map(topic=>({
        id:topic.id,
        title:topic.title,
        excerpts:sources.map(s=>bestEvidence(selectOperativeSourceText(s.data.content),topic))
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
        ...grid.sources.map((s,i)=>"ИСТОЧНИК "+(i+1)+" ("+s.id+") URL: "+s.url+" Название: "+s.title)
    ];
    for(const row of grid.criteria){
        header.push("КРИТЕРИЙ: "+row.title);
        row.excerpts.forEach((quotes,sourceIndex)=>{
            header.push(grid.sources[sourceIndex].id+": "+(quotes.length?
                quotes.slice(0,row.id==="notices"?3:row.id==="patents"?2:1).map(quote=>"«"+
                    quote.replace(/\s+/g," ").slice(0,["use","distribution","restrictions","notices"].includes(row.id)?420:280)+"»").join("; "):
                "Не удалось найти подтверждающую выдержку по этому критерию."));
        });
    }
    header.push("ЮРИДИЧЕСКИЙ СМЫСЛ: выводи из действия в цитате (grants, may, must, does not grant, neither), а не из названия раздела.");
    header.push("Не повторяй только «описано в пункте», «текст определяет» или «указано в лицензии»: объясни конкретное разрешение, требование или запрет.");
    header.push("В таблице обязательства по сохранению уведомлений не заменяй разрешением добавить собственный copyright; право использовать не приравнивай к обязанности.");
    header.push("Если сохранилось условие if/when или provided that — не превращай его в безусловную обязанность.");
    header.push("Ссылки вида S1-use-1 — служебные метки выдержек; изложи содержание своими словами и дай прямые ссылки на первоисточники.");
    header.push("Ответ строй из приведённых выдержек; если они не доказывают вывод, обозначь его как непроверенный. Не делай отрицательных выводов из отсутствия цитаты.");
    return header.join("\n");
}
