import {createHash} from "node:crypto";

const digest = content => createHash("sha256").update(content,"utf8").digest("hex");

/**
 * Translate locally verified references into absolute source offsets.
 * Never store the original document text or unverified model claims here.
 */
export function locateCitedDocumentFragments(window,bundle,checked) {
    if(!window||typeof window.text!=="string"||!bundle?.exactSourceCoverage||
       !(bundle.index instanceof Map)||checked?.success!==true||
       !Array.isArray(checked.citedRefs)||!checked.citedRefs.length)
        return null;
    if(!Number.isSafeInteger(window.number)||window.number<1||
       !Number.isSafeInteger(window.sectionIndex)||window.sectionIndex<0||
       !Number.isSafeInteger(window.startOffset)||window.startOffset<0||
       !Number.isSafeInteger(window.endOffset)||
       window.endOffset!==window.startOffset+window.text.length)
        return null;
    const cited=[...new Set(checked.citedRefs)];
    if(cited.length>18)return null;
    const spans=[];
    for(const localRef of cited){
        if(typeof localRef!=="string"||!/^Ф1:Ф\d{1,3}$/u.test(localRef))return null;
        const fragment=bundle.index.get(localRef);
        if(!fragment||typeof fragment.text!=="string"||
           !Number.isSafeInteger(fragment.startOffset)||
           !Number.isSafeInteger(fragment.endOffset)||
           fragment.startOffset<0||
           fragment.endOffset>window.text.length||
           fragment.endOffset<=fragment.startOffset||
           window.text.slice(fragment.startOffset,fragment.endOffset)!==fragment.text)
            return null;
        spans.push({
            ref:`О${window.number}:${localRef}`,
            sectionIndex:window.sectionIndex,
            sectionTitle:String(window.title||"").slice(0,180),
            windowNumber:window.number,
            startOffset:window.startOffset+fragment.startOffset,
            endOffset:window.startOffset+fragment.endOffset,
            fragmentSha256:digest(fragment.text)
        });
    }
    return spans;
}

/** Distinguish equal filenames and repeated local refs across source files. */
export function scopeCitedDocumentFragments(spans,fileIndex) {
    if(!Number.isSafeInteger(fileIndex)||fileIndex<1||fileIndex>3||
       !Array.isArray(spans)||spans.some(s=>!/^О\d+:Ф1:Ф\d+$/u.test(s?.ref||"")))
       return null;
    return spans.map(s=>({...s,fileIndex,ref:`Д${fileIndex}:${s.ref}`}));
}
