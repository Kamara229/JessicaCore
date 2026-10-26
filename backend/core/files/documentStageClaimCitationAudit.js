import {createLosslessWindowEvidence} from "./documentWindowEvidence.js";

const CLAIM=/^• (?:В документе указано|Позиция автора|Аналитический вывод|Требует независимой проверки): [^\n]+$/gmu;
const REFERENCE=/\[О(\d{1,3}):Ф1:Ф([1-9]\d{0,2})\]/gu;
const EVIDENCE=/^  Основание: ([^\n]+)$/gmu;
const QUOTE=/«([^»]{1,260})»/gu;
const normalize=text=>String(text).normalize("NFKC")
    .replace(/[\u00a0\s]+/gu," ")
    .replace(/[«»„“”]/gu,'\"')
    .replace(/[‐‑‒–—−]/gu,"-").trim().toLocaleLowerCase("ru");

/** Every rendered claim citation must resolve to a fragment holding its quote. */
export function auditStageClaimFragmentLinks(body,sourceText,windowNumber){
    if(typeof body!=="string"||!body||
       !Number.isSafeInteger(windowNumber)||windowNumber<1||
       windowNumber>256)return null;
    const bundle=createLosslessWindowEvidence({name:"source",text:sourceText});
    if(!bundle?.exactSourceCoverage)return null;
    const claims=[...body.matchAll(CLAIM)];
    if(!claims.length||claims.length>14)return null;
    let refs=0,quotes=0;
    for(let i=0;i<claims.length;i++){
        const claim=claims[i],start=claim.index;
        const end=i+1<claims.length?claims[i+1].index:body.length;
        const span=body.slice(start,end);
        const firstLine=span.slice(0,span.indexOf("\n")<0?span.length:span.indexOf("\n"));
        const cited=[...firstLine.matchAll(REFERENCE)];
        const evidence=[...span.matchAll(EVIDENCE)];
        if(!cited.length||cited.length>3||evidence.length!==1||
           cited.some(r=>Number(r[1])!==windowNumber))return null;
        const unique=[...new Set(cited.map(r=>r[2]))];
        if(unique.length!==cited.length)return null;
        const fragments=unique.map(n=>bundle.index.get("Ф1:Ф"+n));
        if(fragments.some(x=>!x))return null;
        const quoted=[...evidence[0][1].matchAll(QUOTE)];
        if(!quoted.length||quoted.length>3)return null;
        const normalized=fragments.map(f=>normalize(f.text));
        const seen=new Set();
        for(const q of quoted){
            const value=normalize(q[1]);
            if(value.length<6)return null;
            let matched=false;
            for(let j=0;j<normalized.length;j++){
                if(normalized[j].includes(value)){seen.add(j);matched=true;}
            }
            if(!matched)return null;
        }
        if(seen.size!==fragments.length)return null;
        refs+=cited.length;quotes+=quoted.length;
    }
    return {claims:claims.length,refs,quotes,fragmentCitationsChecked:true,
        sourceTruthVerified:false,reportAuthenticityVerified:false};
}
