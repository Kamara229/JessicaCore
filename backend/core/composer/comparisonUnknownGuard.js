import { buildComparisonEvidenceGrid } from "./comparisonEvidenceGrid.js";

/*
 * Deterministic presentation safety net. A missing topical excerpt is
 * NOT proof that a document lacks the provision. Only normalize cells
 * for which the retrieved source contains no relevant quotation.
 * This never promotes an answer to "verified": claim validation still
 * checks the complete final text against both original sources.
 */
const UNKNOWN = "Не удалось подтвердить по прочитанному тексту";
const NEGATION = /(?:не\s+(?:содерж\p{L}*|име\p{L}*|предусматр\p{L}*|включа\p{L}*|предоставл\p{L}*)|(?:нет|отсутств\p{L}*)\s+(?:\p{L}+\s+){0,3}(?:положени\p{L}*|услови\p{L}*|патент\p{L}*|лицензи\p{L}*|прав\p{L}*)|does\s+not\s+(?:contain|include|grant|provide)|no\s+(?:patent|license|provision|requirement))/iu;
const TOPICS = {
    use:/использ|измен|модифиц|reproduc|modif|use/iu,
    distribution:/распростран|distribut/iu,
    notices:/уведомлен|авторств|авторск|copyright|notic|attribut/iu,
    patents:/патент|patent/iu,
    restrictions:/огранич|товарн|бренд|имени|назван|trademark|endorse|restrict/iu,
    warranty:/гаранти|ответствен|warrant|liabilit/iu,
    obligations:/обязанност|услови|требован|condition|obligation/iu
};
const SKIP = new Set(["license","licenses","source","document","official","version","opensource",
    "org","com","www","http","https","text","html","clause","license2","creative","initiative"]);

function words(text) {
    return (String(text||"").toLowerCase().match(/[\p{L}\p{N}]+/gu)||[])
        .filter(word=>word.length>=3 && !SKIP.has(word) && !/^\d+$/.test(word));
}
function sourceTerms(source) {
    let url="";
    try {
        const parsed=new URL(source.url);
        url=parsed.hostname+" "+parsed.pathname;
    } catch { /* Unknown source cannot be mapped safely. */ }
    return new Set(words(String(source.title||"")+" "+url));
}
function tableCells(line) {
    const raw=String(line||"").trim();
    if(!raw.startsWith("|") || !raw.endsWith("|")) return null;
    return raw.slice(1,-1).split("|").map(x=>x.trim());
}
function sourceColumnMap(header, sources) {
    if(!header || header.length<3) return null;
    const terms=sources.map(sourceTerms);
    const unique=terms.map((own,i)=>[...own].filter(w=>!terms[1-i].has(w)));
    const used=new Set();
    const indexes=[];
    for(let i=0;i<2;i++) {
        const scored=header.slice(1).map((cell,j)=>({
            col:j+1,score:unique[i].reduce((n,w)=>n+(words(cell).includes(w)?1:0),0)
        })).sort((a,b)=>b.score-a.score);
        if(!scored.length || scored[0].score===0
            || (scored[1] && scored[0].score===scored[1].score)
            || used.has(scored[0].col)) return null;
        indexes[i]=scored[0].col;
        used.add(scored[0].col);
    }
    return indexes;
}
function sourceMention(text,source,other) {
    const own=[...sourceTerms(source)];
    const otherSet=sourceTerms(other);
    const distinctive=own.filter(w=>!otherSet.has(w));
    const present=new Set(words(text));
    return distinctive.some(w=>present.has(w));
}
function missingTopic(row) {
    return row.excerpts.map((quotes,index)=>quotes.length===0?index:-1).filter(i=>i>=0);
}

/*
 * Do not try to guess table column order: identify both columns from
 * document-specific words in their headings. If that fails, leave the
 * answer to the normal semantic validator and its safe partial fallback.
 */
export function guardComparisonUnknowns(task,plan,run,text) {
    if(plan?.evidence?.mode!=="source_content") return {text,changes:0};
    const grid=buildComparisonEvidenceGrid(task,run);
    if(!grid || grid.sources.length!==2) return {text,changes:0};
    const rows=grid.criteria.filter(row=>missingTopic(row).length);
    if(!rows.length) return {text,changes:0};

    const lines=String(text||"").split("\n");
    let mapping=null;
    let changes=0;
    let inTable=false;
    const output=lines.map(line=>{
        const cells=tableCells(line);
        if(cells && !/^[-:\s]+$/.test(cells[0])) {
            if(!inTable) {
                mapping=sourceColumnMap(cells,grid.sources);
                inTable=true;
                return line;
            }
            if(!mapping) return line;
            const row=rows.find(r=>TOPICS[r.id]?.test(cells[0]));
            if(!row) return line;
            for(const i of missingTopic(row)) {
                const col=mapping[i];
                if(col>=cells.length || /не удалось|не установлено|not verified|unable to verify/iu.test(cells[col]))continue;
                cells[col]=UNKNOWN;
                changes++;
            }
            return "| "+cells.join(" | ")+" |";
        }
        if(cells) return line; // Preserve Markdown separators.
        if(!line.trim()) {inTable=false;return line;}
        for(const row of rows) {
            if(!TOPICS[row.id]?.test(line)||!NEGATION.test(line))continue;
            const index=missingTopic(row).find(i=>
                sourceMention(line,grid.sources[i],grid.sources[1-i]));
            if(index===undefined)continue;
            changes++;
            return "По критерию «"+row.title+"» для источника "+
                grid.sources[index].url+" — "+UNKNOWN.toLowerCase()+".";
        }
        return line;
    });
    return {text:output.join("\n"),changes};
}
