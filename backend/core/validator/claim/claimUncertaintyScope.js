/*
 * A statement that a provision COULD NOT BE VERIFIED is a report of
 * uncertainty, not an assertion that the provision does not exist.
 * A validation model sometimes invents a negative claim for such a
 * table cell and marks it unsupported. Ignore that specific phantom
 * claim only when the final answer has an explicitly unverified cell
 * for the SAME source and topic. Never suppress actual categorical
 * negatives or claims about another source.
 */
const UNKNOWN=/не удалось (?:подтвердить|проверить|установить)|не (?:подтвержден[оаы]?|подтверждается|установлено)|нет подтверждения|not verified|unable to verify|could not confirm/i;
const CATEGORICAL=/не содержит|не имеет|отсутств(?:ует|уют)|does not contain|does not include|no (?:patent|provision|requirement)/i;
const STOP=new Set(["www","http","https","org","com","html","license","licenses","clause","source","text","opensource","version"]);
const TOPICS=[
    /патент|patent/i,
    /уведомлен|копирайт|авторск|copyright|notice|attribut/i,
    /распростран|дистрибуц|distribut/i,
    /огранич|товарн|имен|назван|trademark|endors|restrict/i,
    /использован|модифиц|изменени|reproduc|modif|permission|use/i
];
function tokens(str) {
    return (String(str||"").toLowerCase().match(/[a-z]{3,}/g)||[])
        .filter(token=>!STOP.has(token));
}
function sourceIdentifiers(a,b) {
    const other=new Set(tokens(b));
    return [...new Set(tokens(a))].filter(token=>!other.has(token));
}
function cells(line) {
    const txt=String(line||"").trim();
    return txt.startsWith("|")&&txt.endsWith("|")
        ?txt.slice(1,-1).split("|").map(x=>x.trim()) : null;
}
export function isUnassertedUncertaintyClaim(claim,answer,sourceUrls=[]) {
    const statement=String(claim?.claim||"");
    if(!UNKNOWN.test(statement)||CATEGORICAL.test(statement))return false;
    if(!Array.isArray(sourceUrls)||sourceUrls.length!==2)return false;
    const sourceIndex=sourceUrls.findIndex(url=>url===claim?.sourceUrl);
    if(sourceIndex<0)return false;
    const topic=TOPICS.find(re=>re.test(statement));
    if(!topic)return false;
    const identifiers=sourceIdentifiers(sourceUrls[sourceIndex],sourceUrls[1-sourceIndex]);
    if(!identifiers.length)return false;
    const lines=String(answer||"").split("\n");
    for(let i=0;i<lines.length;i++) {
        const header=cells(lines[i]);
        if(!header||header.length<3||!/критер|услови|aspect|criteria|criterion/i.test(header[0]))continue;
        const indexes=header.map((value,j)=>j>0 && tokens(value).some(t=>identifiers.includes(t))?j:-1)
            .filter(j=>j>0);
        if(indexes.length!==1)continue;
        const col=indexes[0];
        for(let n=i+1;n<lines.length;n++) {
            const row=cells(lines[n]);
            if(!row)break;
            if(!topic.test(row[0]||""))continue;
            const value=row[col]||"";
            if(UNKNOWN.test(value)&&!CATEGORICAL.test(value))return true;
        }
    }
    return false;
}
export function partitionComparisonClaims(claims,answer,sourceUrls) {
    const kept=[],skipped=[];
    for(const item of Array.isArray(claims)?claims:[]) {
        (isUnassertedUncertaintyClaim(item,answer,sourceUrls)?skipped:kept).push(item);
    }
    return {claims:kept,ignoredUncertainty:skipped.length};
}
