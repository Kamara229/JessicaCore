import {createHash} from "node:crypto";
const digest=s=>createHash("sha256").update(s,"utf8").digest("hex");

/** Number alignment only: never certifies citation relevance or source existence. */
export function linkNumberedDocumentReferences(source,index,mentions,entries,hasBibliography) {
    if(typeof source!=="string"||!index?.complete||
       !Array.isArray(index.sections)||!Array.isArray(mentions)||
       !Array.isArray(entries)||mentions.length>600||entries.length>600)
        return null;
    const locations=new Map();
    for(const entry of entries){
        if(!Number.isSafeInteger(entry.number)||entry.number<1||
           !Number.isSafeInteger(entry.startOffset)||
           !Number.isSafeInteger(entry.endOffset)||
           entry.startOffset<0||entry.endOffset>source.length||
           entry.endOffset<=entry.startOffset||
           digest(source.slice(entry.startOffset,entry.endOffset))!==entry.lineSha256)
            return null;
        if(!locations.has(entry.number))locations.set(entry.number,[]);
        locations.get(entry.number).push(entry.startOffset);
    }
    const links=[];
    for(const mention of mentions){
        const section=index.sections[mention.sectionIndex];
        if(!section||!Number.isSafeInteger(mention.number)||
           !Number.isSafeInteger(mention.startOffset)||
           !Number.isSafeInteger(mention.endOffset)||
           mention.startOffset<section.startOffset||
           mention.endOffset>section.endOffset||
           mention.endOffset<=mention.startOffset||
           digest(source.slice(mention.startOffset,mention.endOffset))!==
               mention.literalSha256)return null;
        const possible=locations.get(mention.number)||[];
        const match= !hasBibliography?"no-bibliography":
            possible.length===0?"missing-entry":
            possible.length>1?"duplicate-number":"number-match";
        links.push({number:mention.number,sectionIndex:mention.sectionIndex,
            sectionTitle:section.title,sourceStart:mention.startOffset,
            sourceEnd:mention.endOffset,literalSha256:mention.literalSha256,
            match,entryStartOffset:match==="number-match"?possible[0]:null});
    }
    return {mentions:links,numberMatched:links.filter(x=>x.match==="number-match").length,
        ambiguous:links.filter(x=>x.match==="duplicate-number").length,
        missing:links.filter(x=>x.match==="missing-entry").length,
        bibliographyAbsent:!hasBibliography,
        sourceRelevanceVerified:false,sourceExistenceVerified:false,
        pageAccuracyVerified:false};
}
