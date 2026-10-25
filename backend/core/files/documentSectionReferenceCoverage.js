const RELATIONS=new Set(["number-match","missing-entry","duplicate-number","no-bibliography"]);
const numeric=(n)=>Number.isSafeInteger(n)&&n>=0;

/**
 * Count literal numbered-reference candidates in each source-derived heading
 * and its descendants. This is not a relevance, authenticity or legal audit.
 */
export function summarizeSectionReferenceCoverage(referenceAudit, hierarchy) {
    const refs=referenceAudit?.referenceLinks;
    if(!referenceAudit||referenceAudit.completeCandidateScan!==true||
       !refs||!Array.isArray(refs.mentions)||!hierarchy||
       !Array.isArray(hierarchy.sections)||!hierarchy.sections.length||
       hierarchy.headingHierarchyHeuristic!==true||
       hierarchy.sourceChars!==referenceAudit.sourceChars||
       refs.mentions.length!==referenceAudit.citationMentions)return null;
    const rows=[],lookup=new Map();
    let cursor=0;
    for(const section of hierarchy.sections){
        if(!numeric(section?.sectionIndex)||section.sectionIndex!==rows.length||
           !numeric(section.startOffset)||section.startOffset!==cursor||
           !numeric(section.endOffset)||section.endOffset<=cursor||
           typeof section.title!=="string"||
           !Array.isArray(section.headingPath)||!section.headingPath.length||
           section.headingPath.at(-1)!==section.title||
           !numeric(section.rootSectionIndex)||
           (section.parentSectionIndex!==null&&
            (!numeric(section.parentSectionIndex)||
             section.parentSectionIndex>=section.sectionIndex)))
            return null;
        if(section.parentSectionIndex!==null){
            const parent=lookup.get(section.parentSectionIndex);
            if(!parent||section.rootSectionIndex!==parent.rootSectionIndex||
               section.headingPath.length!==parent.headingPath.length+1||
               parent.headingPath.some((label,i)=>label!==section.headingPath[i]))
                return null;
        }else if(section.rootSectionIndex!==section.sectionIndex)return null;
        const item={sectionIndex:section.sectionIndex,
            title:section.title,parentSectionIndex:section.parentSectionIndex,
            rootSectionIndex:section.rootSectionIndex,
            headingPath:[...section.headingPath],
            citationMentionsDirect:0,citationMentionsSubtree:0,
            matchedDirect:0,matchedSubtree:0,
            missingDirect:0,missingSubtree:0,
            unresolvedDirect:0,unresolvedSubtree:0};
        rows.push(item);
        lookup.set(item.sectionIndex,item);
        cursor=section.endOffset;
    }
    if(cursor!==referenceAudit.sourceChars)return null;
    let matched=0,missing=0,unresolved=0;
    for(const citation of refs.mentions){
        if(!numeric(citation?.sectionIndex)||
           !lookup.has(citation.sectionIndex)||
           !RELATIONS.has(citation.match)||
           !numeric(citation.number)||citation.number<1||
           !numeric(citation.sourceStart)||!numeric(citation.sourceEnd)||
           citation.sourceEnd<=citation.sourceStart)return null;
        const origin=hierarchy.sections[citation.sectionIndex];
        if(citation.sourceStart<origin.startOffset||
           citation.sourceEnd>origin.endOffset)return null;
        const key=citation.match==="number-match"?"matched":
           citation.match==="missing-entry"?"missing":"unresolved";
        if(key==="matched")matched++;
        else if(key==="missing")missing++;
        else unresolved++;
        const direct=lookup.get(citation.sectionIndex);
        direct.citationMentionsDirect++;
        direct[key+"Direct"]++;
        let current=direct,steps=0;
        while(current){
            if(++steps>rows.length)return null;
            current.citationMentionsSubtree++;
            current[key+"Subtree"]++;
            current=current.parentSectionIndex===null?null:
                lookup.get(current.parentSectionIndex);
            if(current===undefined)return null;
        }
    }
    if(matched!==refs.numberMatched||missing!==refs.missing||
       unresolved!==refs.mentions.length-matched-missing)return null;
    return {sourceChars:referenceAudit.sourceChars,
        citationMentions:refs.mentions.length,matched,missing,unresolved,
        sections:rows,
        sectionsWithDirectCitations:rows.filter(x=>x.citationMentionsDirect>0).length,
        sectionsWithDescendantCitations:rows.filter(x=>x.citationMentionsSubtree>0).length,
        bibliographyAbsent:refs.bibliographyAbsent,
        syntacticReferencesOnly:true,
        academicSourceQualityVerified:false};
}
