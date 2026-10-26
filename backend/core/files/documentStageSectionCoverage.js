import {buildDocumentSectionIndex} from "./documentSectionIndex.js";

/**
 * A source-derived section map for validated stage REPORT metadata.
 * Unsigned reports never count as authenticated analysis of a section.
 * Section headings are navigation hints, not proof of scholarly meaning.
 */
export function summarizeDocumentStageSectionCoverage(source,verified,reportedWindows,authenticated){
    if(typeof source!=="string"||!source.length||
       !Array.isArray(verified?.windows)||!verified.windows.length||
       !Number.isSafeInteger(reportedWindows)||reportedWindows<0||
       reportedWindows>verified.windows.length||
       typeof authenticated!=="boolean")return null;
    const index=buildDocumentSectionIndex(source);
    if(!index.complete||index.sourceChars!==source.length||
       !Array.isArray(index.sections)||!index.sections.length)return null;
    const sections=index.sections.map((section,i)=>({
        sectionIndex:i,title:String(section.title||"").slice(0,180),
        startOffset:section.startOffset,endOffset:section.endOffset,
        plannedWindows:0,reportedWindows:0,authenticatedWindows:0,
        allSectionWindowsAuthenticated:false
    }));
    let cursor=0;
    for(let i=0;i<verified.windows.length;i++){
        const w=verified.windows[i],section=sections[w?.sectionIndex];
        if(!section||w.number!==i+1||
           !Number.isSafeInteger(w.startOffset)||
           w.startOffset!==cursor||
           !Number.isSafeInteger(w.endOffset)||w.endOffset<=cursor||
           w.endOffset>section.endOffset||
           w.startOffset<section.startOffset)return null;
        section.plannedWindows++;
        if(i<reportedWindows){
            section.reportedWindows++;
            if(authenticated)section.authenticatedWindows++;
        }
        cursor=w.endOffset;
    }
    if(cursor!==source.length||
       sections.some(s=>s.plannedWindows===0))return null;
    for(const section of sections)
        section.allSectionWindowsAuthenticated=
            section.authenticatedWindows===section.plannedWindows;
    return {sections,sectionCount:sections.length,
        reportedSections:sections.filter(s=>s.reportedWindows>0).length,
        authenticatedSections:sections.filter(s=>s.allSectionWindowsAuthenticated).length,
        allSectionsAuthenticated:sections.every(s=>s.allSectionWindowsAuthenticated),
        originalSectionIndexComplete:true,
        scientificCorrectnessVerified:false};
}
