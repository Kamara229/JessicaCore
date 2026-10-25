import {buildDocumentSectionIndex} from "./documentSectionIndex.js";

/**
 * Reconstruct the heuristic heading tree only from the cryptographically
 * verified source plan; never from claims in the model's response.
 * Root headings, preface, introduction and conclusion cannot become parents.
 */
export function buildVerifiedSectionHierarchy(plan, auditedSections) {
    if(plan?.complete!==true||!Array.isArray(plan.windows)||
       !plan.windows.length||plan.windows.length>16||
       !Array.isArray(auditedSections))return null;
    const source=plan.windows.map(w=>w?.text).join("");
    if(source.length!==plan.sourceChars)return null;
    const index=buildDocumentSectionIndex(source);
    if(!index.complete||index.sections.length!==auditedSections.length)
        return null;
    const roots=[],stack=[],sections=[];
    let expectedStart=0;
    for(let i=0;i<index.sections.length;i++){
        const observed=index.sections[i],record=auditedSections[i];
        if(!record||record.sectionIndex!==i||
           record.title!==observed.title||
           record.startOffset!==observed.startOffset||
           record.endOffset!==observed.endOffset||
           record.startOffset!==expectedStart||
           !Number.isSafeInteger(observed.level)||
           observed.level<0||observed.level>4||
           record.plannedWindows<1||
           record.processedWindows<0||
           record.processedWindows>record.plannedWindows)return null;
        expectedStart=observed.endOffset;
        // A heading at level 0 or 1 starts a new independent root.
        if(observed.level<=1)stack.length=0;
        else while(stack.length && stack.at(-1).level>=observed.level)
            stack.pop();
        const ancestor=observed.level>1&&stack.length
            ?stack.at(-1):null;
        const path=ancestor?[...ancestor.path,observed.title]:
            [observed.title];
        const entry={sectionIndex:i,title:observed.title,level:observed.level,
            parentSectionIndex:ancestor?.sectionIndex??null,
            rootSectionIndex:ancestor?.rootSectionIndex??i,
            headingPath:path,depth:path.length-1,
            startOffset:observed.startOffset,
            endOffset:observed.endOffset,
            processedWindows:record.processedWindows,
            plannedWindows:record.plannedWindows,
            complete:record.processedWindows===record.plannedWindows};
        sections.push(entry);
        if(!ancestor)roots.push(i);
        if(observed.level>=1)stack.push(entry);
    }
    if(expectedStart!==plan.sourceChars)return null;
    return {sections,roots,headingHierarchyHeuristic:true,
        sourceChars:expectedStart,
        fullyProcessed:sections.every(s=>s.complete),
        semanticStructureConfirmed:false};
}
