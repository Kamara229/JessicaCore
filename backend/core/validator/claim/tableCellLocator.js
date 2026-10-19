/*
 * Locate a claim's actual comparison-table cell.
 * Parse source identity from both source URLs and table headings.
 * Never guess a column when mapping is ambiguous.
 */
const OMIT=new Set(["http","https","www","org","com","html","txt","license",
    "licenses","opensource","source","version","terms"]);
const TOPICS=[
    /патент|patent/iu,
    /уведомлен|авторск|копирайт|copyright|notice|attribut/iu,
    /распростран|дистрибуц|distribut/iu,
    /огранич|товарн|имени|назван|trademark|endors|restrict/iu,
    /использован|модифиц|изменен|reproduc|modif|permission|use/iu
];

export function tableTokens(value) {
    return (String(value||"").toLowerCase().match(/[a-z0-9]{3,}/g)||[])
        .filter(term=>!OMIT.has(term));
}

function cells(line) {
    const value=String(line||"").trim();
    return value.startsWith("|")&&value.endsWith("|")
        ?value.slice(1,-1).split("|").map(x=>x.trim()) : null;
}
function isSeparator(items) {
    return items&&items.every(x=>/^:?-{2,}:?$/.test(x));
}
function uniqueIdentifiers(url,otherUrl) {
    const other=new Set(tableTokens(otherUrl));
    return [...new Set(tableTokens(url))].filter(x=>!other.has(x));
}
function matchColumn(header,terms) {
    const indexes=[];
    for(let index=1;index<header.length;index++) {
        if(tableTokens(header[index]).some(x=>terms.includes(x)))indexes.push(index);
    }
    return indexes.length===1?indexes[0]:-1;
}
export function locateComparisonCell({answer,sourceUrls,claim}={}) {
    if(!Array.isArray(sourceUrls)||sourceUrls.length!==2)return null;
    const sourceIndex=sourceUrls.indexOf(String(claim?.sourceUrl||""));
    if(sourceIndex<0)return null;
    const topic=TOPICS.find(re=>re.test(String(claim?.claim||"")));
    if(!topic)return null;
    const first=uniqueIdentifiers(sourceUrls[0],sourceUrls[1]);
    const second=uniqueIdentifiers(sourceUrls[1],sourceUrls[0]);
    if(!first.length||!second.length)return null;

    let columns=null;
    for(const line of String(answer||"").split("\n")) {
        const parts=cells(line);
        if(!parts) {columns=null;continue;}
        if(isSeparator(parts))continue;
        const proposed=[matchColumn(parts,first),matchColumn(parts,second)];
        if(proposed[0]>0&&proposed[1]>0&&proposed[0]!==proposed[1]) {
            columns=proposed;continue;
        }
        if(!columns)continue;
        if(!topic.test(parts[0]||""))continue;
        const col=columns[sourceIndex];
        if(col>=parts.length)return null;
        return {value:parts[col],criterion:parts[0],sourceIndex,
            column:col,sourceUrl:sourceUrls[sourceIndex]};
    }
    return null;
}
