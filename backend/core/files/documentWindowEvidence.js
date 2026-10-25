import {DOCUMENT_WINDOW_CHARS} from "./documentTextWindow.js";

const FRAGMENT_CHARS=900;

/** Exact UTF-16 spans in one verified source window; no trimming or omission. */
export function createLosslessWindowEvidence({name,text}={}) {
    if(typeof text!=="string"||!text||text.length>DOCUMENT_WINDOW_CHARS)return null;
    const safeName=String(name||"Документ").replace(/[\r\n\t<>]/gu," ").slice(0,100);
    const fragments=[],index=new Map();
    let at=0;
    while(at<text.length){
        let end=Math.min(at+FRAGMENT_CHARS,text.length);
        if(end<text.length&&/[\uD800-\uDBFF]/u.test(text[end-1])&&
           /[\uDC00-\uDFFF]/u.test(text[end]))end--;
        if(end<=at)return null;
        const ref=`Ф1:Ф${fragments.length+1}`;
        const fragment={ref,name:safeName,text:text.slice(at,end),startOffset:at,endOffset:end};
        fragments.push(fragment);index.set(ref,fragment);at=end;
    }
    if(fragments.map(f=>f.text).join("")!==text)return null;
    return {
        docs:[{name:safeName,text,truncated:false}],
        fragments,index,partial:false,omitted:0,exactSourceCoverage:true,
        promptText:fragments.map(f=>
            `[${f.ref}] (${f.name}, символы ${f.startOffset}–${f.endOffset})\n${f.text}`
        ).join("\n\n")
    };
}
