import {locateComparisonCell,tableTokens} from "./tableCellLocator.js";

/*
 * Claim checker must not validate assertions it invented for a cell
 * explicitly marked "unverified". Keep actual facts in the answer
 * (including categorical absence) subject to normal evidence checks.
 * This filter does not turn any claim into supported=true.
 */
const UNKNOWN=/не удалось (?:подтвердить|проверить|установить)|не (?:подтвержден|подтверждается|установлено)|нет подтверждения|not verified|unable to verify|could not confirm/iu;
const ABSENCE=/не\s+(?:содерж|имеет|предусматр|включа|предоставл)|отсутств|does\s+not\s+(?:contain|include|grant)|no\s+(?:patent|requirement|provision)/iu;
const TOPICS=[
    /патент|patent/iu,
    /уведомлен|авторск|копирайт|copyright|notice|attribut/iu,
    /распростран|distribut/iu,
    /огранич|товарн|имени|назван|trademark|endors|restrict/iu,
    /использован|модифиц|изменен|reproduc|modif|permission|use/iu
];

function assertedOutsideTable(answer,claim,urls) {
    const sourceIndex=urls.indexOf(String(claim?.sourceUrl||""));
    if(sourceIndex<0)return true;
    const other=new Set(tableTokens(urls[1-sourceIndex]));
    const identifiers=tableTokens(urls[sourceIndex]).filter(x=>!other.has(x));
    const topic=TOPICS.find(re=>re.test(String(claim?.claim||"")));
    if(!topic||!identifiers.length)return true;

    return String(answer||"").split("\n").some(line=>{
        const trimmed=line.trim();
        if(!trimmed||trimmed.startsWith("|")||UNKNOWN.test(trimmed))return false;
        if(!topic.test(trimmed))return false;
        return tableTokens(trimmed).some(token=>identifiers.includes(token));
    });
}
export function shouldDiscardUnassertedClaim(claim,answer,sourceUrls) {
    const matched=locateComparisonCell({answer,sourceUrls,claim});
    if(!matched)return false;
    if(!UNKNOWN.test(matched.value)||ABSENCE.test(matched.value))return false;
    // Text outside the table may still assert a real positive or
    // negative fact about this same document/topic. Do not hide it.
    if(assertedOutsideTable(answer,claim,sourceUrls))return false;
    return true;
}
export function filterUnassertedUncertaintyClaims(claims,answer,sourceUrls) {
    const source=Array.isArray(claims)?claims:[];
    const result=source.filter(x=>!shouldDiscardUnassertedClaim(x,answer,sourceUrls));
    return {claims:result,ignoredUncertainty:source.length-result.length};
}
