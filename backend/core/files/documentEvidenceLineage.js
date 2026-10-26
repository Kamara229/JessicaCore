import {MAX_VERIFIED_DOCUMENT_WINDOWS} from "./documentVerifiedWindowLimit.js";
const SHA=/^[a-f0-9]{64}$/u;
const LOCAL_REF=/^О(?:[1-9]|[12]\d|3[0-2]):Ф1:Ф[1-9]\d{0,2}$/u;

/**
 * Join quote-verified citation fragments to the source-derived heading tree.
 * Never use a model-generated heading as documentary provenance.
 */
export function buildDocumentEvidenceLineage(
    sectionCoverage, readingCoverage, citationSpans, verifiedClaims, quoteRanges
) {
    const tree=sectionCoverage?.hierarchy;
    const windows=readingCoverage?.verifiedWindows;
    if(tree?.fullyProcessed!==true||tree?.semanticStructureConfirmed!==false||
       !Array.isArray(tree.sections)||!tree.sections.length||
       !SHA.test(String(readingCoverage?.sourceSha256||""))||
       readingCoverage?.allWindowsPresented!==true||
       !Array.isArray(windows)||!windows.length||windows.length>MAX_VERIFIED_DOCUMENT_WINDOWS||
       !Array.isArray(citationSpans)||!citationSpans.length||
       !Array.isArray(verifiedClaims)||!verifiedClaims.length||
       !Array.isArray(quoteRanges))return null;
    const sourceSections=new Map();
    for(const section of tree.sections){
        if(!Number.isSafeInteger(section.sectionIndex)||
           sourceSections.has(section.sectionIndex)||
           !Number.isSafeInteger(section.startOffset)||
           !Number.isSafeInteger(section.endOffset)||
           section.endOffset<=section.startOffset||
           section.complete!==true||
           !Array.isArray(section.headingPath)||!section.headingPath.length||
           section.headingPath.at(-1)!==section.title)
            return null;
        sourceSections.set(section.sectionIndex,section);
    }
    const byWindow=new Map();
    for(const window of windows){
        if(!Number.isSafeInteger(window?.number)||window.number!==byWindow.size+1||
           !Number.isSafeInteger(window.startOffset)||
           !Number.isSafeInteger(window.endOffset)||
           window.endOffset<=window.startOffset||
           !sourceSections.has(window.sectionIndex))return null;
        const section=sourceSections.get(window.sectionIndex);
        if(window.startOffset<section.startOffset||window.endOffset>section.endOffset)
            return null;
        byWindow.set(window.number,window);
    }
    const citations=new Map();
    for(const span of citationSpans){
        const window=byWindow.get(span?.windowNumber);
        const section=sourceSections.get(span?.sectionIndex);
        if(!window||!section||span.sectionIndex!==window.sectionIndex||
           !LOCAL_REF.test(String(span.ref||""))||
           !span.ref.startsWith(`О${window.number}:`)||
           citations.has(span.ref)||
           !Number.isSafeInteger(span.startOffset)||
           !Number.isSafeInteger(span.endOffset)||
           span.startOffset<window.startOffset||span.endOffset>window.endOffset||
           span.endOffset<=span.startOffset||
           span.sectionTitle!==section.title||
           !SHA.test(String(span.fragmentSha256||"")))return null;
        citations.set(span.ref,{ref:span.ref,fileIndex:1,
            sectionIndex:section.sectionIndex,
            rootSectionIndex:section.rootSectionIndex,
            headingPath:[...section.headingPath],
            windowNumber:window.number,startOffset:span.startOffset,
            endOffset:span.endOffset,fragmentSha256:span.fragmentSha256,
            exactQuoteCount:0,verifiedClaimMentions:0});
    }
    let verifiedClaimCount=0;
    for(const claim of verifiedClaims){
        const window=byWindow.get(claim?.windowNumber);
        if(!window||claim.fileIndex!==1||
           !Array.isArray(claim.refs)||!claim.refs.length||claim.refs.length>3)
           return null;
        for(const ref of new Set(claim.refs)){
            const citation=citations.get(ref);
            if(!citation||citation.windowNumber!==window.number||
               citation.sectionIndex!==window.sectionIndex)return null;
            citation.verifiedClaimMentions++;
        }
        verifiedClaimCount++;
    }
    let exactQuoteCount=0;
    for(const quote of quoteRanges){
        const citation=citations.get(quote?.ref);
        if(!citation||quote.windowNumber!==citation.windowNumber||
           quote.sectionIndex!==citation.sectionIndex||
           !Number.isSafeInteger(quote.startOffset)||
           !Number.isSafeInteger(quote.endOffset)||
           quote.startOffset<citation.startOffset||
           quote.endOffset>citation.endOffset||
           quote.endOffset<=quote.startOffset||
           !SHA.test(String(quote.quoteSha256||"")))return null;
        citation.exactQuoteCount++;
        exactQuoteCount++;
    }
    if([...citations.values()].some(row=>row.verifiedClaimMentions<1))return null;
    return {sourceSha256:readingCoverage.sourceSha256,
        sourceChars:readingCoverage.sourceChars,
        citationCount:citations.size,verifiedClaimCount,exactQuoteCount,
        citations:[...citations.values()],
        headingHierarchyHeuristic:true,semanticCorrectnessConfirmed:false};
}

/** Keep identical local refs from different files unambiguous. */
export function scopeDocumentEvidenceLineage(lineage,fileIndex,sourceSha256){
    if(!Number.isSafeInteger(fileIndex)||fileIndex<1||fileIndex>3||
       !lineage||!SHA.test(String(sourceSha256||""))||
       lineage.sourceSha256!==sourceSha256||
       !Array.isArray(lineage.citations)||!lineage.citations.length||
       lineage.citationCount!==lineage.citations.length)return null;
    const citations=[];
    for(const row of lineage.citations){
        if(row.fileIndex!==1||!LOCAL_REF.test(String(row.ref||""))||
           !Array.isArray(row.headingPath)||!row.headingPath.length)return null;
        citations.push({...row,fileIndex,ref:`Д${fileIndex}:${row.ref}`});
    }
    return {fileIndex,sourceSha256,sourceChars:lineage.sourceChars,
        citationCount:lineage.citationCount,
        verifiedClaimCount:lineage.verifiedClaimCount,
        exactQuoteCount:lineage.exactQuoteCount,citations,
        headingHierarchyHeuristic:true,semanticCorrectnessConfirmed:false};
}
