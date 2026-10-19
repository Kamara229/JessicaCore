/*
 * Extract contiguous verbatim citation candidates for duties mentioned
 * in a comparison. This is retrieval, NOT verification of legal meaning.
 * Keep a short conditional prefix so "if/when" is not lost.
 */
const NOTICE_QUESTION=/уведомлен|авторск|копирайт|copyright|notice|attribut|сохранен/iu;
const DUTY=/\bmust\s+(?:retain|reproduce|give|include|cause|provide)\b|\bshall\s+(?:retain|include|provide)\b|\brequired\s+to\s+(?:retain|include|provide)\b/giu;
const CONDITIONAL=/\bif\s+the\s+work\s+includes\s+a\s+.{0,35}notice\b/giu;
function excerptAround(text,offset) {
    const start=Math.max(0,offset-85);
    const end=Math.min(text.length,offset+440);
    return text.slice(start,end).trim();
}
export function extractNoticeDutyExcerpts(document,task) {
    if(!NOTICE_QUESTION.test(String(task||"")))return [];
    const text=String(document||"").replace(/\s+/g," ").slice(0,30000);
    const found=[];
    for(const [pattern,bonus] of [[DUTY,0],[CONDITIONAL,18]]) {
        pattern.lastIndex=0;
        for(const match of text.matchAll(pattern)) {
            const quote=excerptAround(text,match.index);
            if(!/copyright|notice|attribut|license/i.test(quote))continue;
            let score=bonus;
            if(/\bmust\s+retain\b/i.test(quote))score+=30;
            if(/\bif\b.{0,100}\bnotice\b/i.test(quote))score+=20;
            if(/\bmust\s+(?:give|include|reproduce)\b/i.test(quote))score+=15;
            if(/copyright\s*\[yyyy\]/i.test(quote)&&!/\bmust\b/i.test(quote))score-=80;
            found.push({offset:match.index,quote,score});
            if(found.length>=20)break;
        }
    }
    found.sort((a,b)=>b.score-a.score||a.offset-b.offset);
    const selected=[];
    for(const entry of found) {
        if(selected.some(x=>Math.abs(x.offset-entry.offset)<230))continue;
        selected.push(entry);
        if(selected.length===3)break;
    }
    return selected.map(x=>x.quote);
}
