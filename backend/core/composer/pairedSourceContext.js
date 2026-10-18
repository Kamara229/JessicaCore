/*
 * Bounded evidence windows for comparisons of two retrieved documents.
 * Keeps verbatim excerpts from EACH fetched source; no summarization
 * or factual inference is performed here.
 */
const CHUNK=800;
const STEP=680;
const PER_SOURCE=3600;

function tokensFromTask(task) {
    const text=String(task || "").toLowerCase();
    const terms=(text.match(/[a-z]{4,}/g)||[]).filter(t=>!["https","http","license","source"].includes(t));
    const aliases=[
        [/патент/i,["patent","grant"]],
        [/прав[ао]|разреш/i,["permission","rights","grant"]],
        [/измен|модифиц/i,["modified","modification","change","derivative"]],
        [/распростран/i,["distribution","distribute"]],
        [/уведомлен|копирайт/i,["notice","copyright"]]
    ];
    for(const [test,words] of aliases) if(test.test(text))terms.push(...words);
    return [...new Set(terms)];
}

function windows(content,terms) {
    const str=String(content||"").trim();
    if(!str)return [];
    if(str.length<=PER_SOURCE)return [str];

    const chunks=[];
    for(let i=0;i<str.length;i+=STEP) {
        const snippet=str.slice(i,i+CHUNK);
        const low=snippet.toLowerCase();
        const score=terms.reduce((sum,word)=>sum+(low.includes(word)?2:0),0);
        chunks.push({i,snippet,score});
        if(i+CHUNK>=str.length)break;
    }
    const first=chunks[0];
    const ranked=chunks.slice(1).sort((a,b)=>b.score-a.score||a.i-b.i);
    const chosen=[first,...ranked.slice(0,3)].sort((a,b)=>a.i-b.i);
    return chosen.map(({i,snippet})=>"[Отрывок с позиции "+i+"]\n"+snippet);
}

export function buildPairedSourceContext(task,taskRunResult) {
    const results=(taskRunResult?.results||[]).filter(r=>r?.tool==="web_fetch");
    if(results.length!==2||results.some(r=>r?.success!==true||!r?.data?.url||!r?.data?.content))return null;
    const terms=tokensFromTask(task);
    return results.map((r,i)=>{
        const d=r.data;
        return ["ИСТОЧНИК "+(i+1), "URL: "+String(d.url),
            "Название: "+String(d.title||""),
            "Ниже приведены дословные выдержки, а не полный документ.",
            ...windows(d.content,terms)].join("\n").slice(0,PER_SOURCE+360);
    }).join("\n\n--------------------\n\n");
}
