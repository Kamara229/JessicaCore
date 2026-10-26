/** Structural window/section continuity. Does not infer semantic coherence. */
export function inspectStageClaimWindowContinuity(selected,verified){
    if(!Array.isArray(selected)||!selected.length||selected.length>768||
       verified?.complete!==true||verified.allSourceWindowsVerified!==true||
       !Array.isArray(verified.windows)||!verified.windows.length||
       verified.windows.length!==verified.totalWindows||
       verified.windows.length>1024)return null;
    let previousEnd=0,previousSection=null,sectionTransitions=0;
    let sectionOrderChanges=0;
    const transitions=[],allSections=new Set();
    for(const [index,w] of verified.windows.entries()){
        if(w?.number!==index+1||
           !Number.isSafeInteger(w.sectionIndex)||w.sectionIndex<0||
           !Number.isSafeInteger(w.startOffset)||
           !Number.isSafeInteger(w.endOffset)||
           w.startOffset!==previousEnd||w.endOffset<=w.startOffset)
            return null;
        if(previousSection!==null&&w.sectionIndex!==previousSection){
            sectionTransitions++;
            if(w.sectionIndex<previousSection)sectionOrderChanges++;
            transitions.push({fromWindow:index,toWindow:index+1,
                fromSection:previousSection,toSection:w.sectionIndex});
        }
        allSections.add(w.sectionIndex);
        previousEnd=w.endOffset;
        previousSection=w.sectionIndex;
    }
    if(previousEnd!==verified.sourceChars)return null;
    const selectedWindows=new Set(),selectedSections=new Set();
    let previousWindow=0;
    for(const c of selected){
        const w=verified.windows[c?.windowNumber-1];
        if(!w||!Number.isSafeInteger(c.windowNumber)||
           c.windowNumber<previousWindow||
           c.windowNumber!==w.number||
           c.sectionIndex!==w.sectionIndex||
           c.startOffset!==w.startOffset||c.endOffset!==w.endOffset)
            return null;
        previousWindow=c.windowNumber;
        selectedWindows.add(c.windowNumber);
        selectedSections.add(c.sectionIndex);
    }
    const order=[...selectedWindows];
    let skippedBetweenSelections=0;
    for(let i=1;i<order.length;i++)
        skippedBetweenSelections+=order[i]-order[i-1]-1;
    return {sourceWindowsContiguous:true,
        selectedClaimsLocated:selected.length,
        selectedWindowsLocated:order.length,
        sourceWindowCount:verified.totalWindows,
        sourceSectionCount:allSections.size,
        selectedSectionCount:selectedSections.size,
        sectionsWithoutSelectedClaims:allSections.size-selectedSections.size,
        windowsWithoutSelectedClaims:verified.totalWindows-order.length,
        gapsBetweenSelectedWindows:skippedBetweenSelections,
        sectionTransitions,sectionOrderChanges,
        sectionTransitionsLocated:transitions,
        sectionStructureAudited:true,
        semanticSectionCoherenceVerified:false,
        completeDocumentInterpretationVerified:false};
}
