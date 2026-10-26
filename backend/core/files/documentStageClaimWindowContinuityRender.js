/** Compact outline audit; source quotations remain in the signed claim renderer. */
export function renderStageClaimWindowContinuity(outline,crossPage){
    if(outline?.sourceWindowsContiguous!==true||
       outline?.sectionStructureAudited!==true||
       !Array.isArray(outline.sectionTransitionsLocated)||
       outline.sectionTransitionsLocated.length>1024||
       crossPage?.crossPageCandidateAnchorsVerified!==true||
       !Array.isArray(crossPage.crossPageCandidateLocations)||
       crossPage.crossPageCandidateLocations.length>12)return null;
    const summary="\n\nПроверка структуры длинного документа: "+
        "проверено исходных окон "+outline.sourceWindowCount+
        ", разделов "+outline.sourceSectionCount+
        "; выбранных окон "+outline.selectedWindowsLocated+
        ", разделов с выбранными утверждениями "+
        outline.selectedSectionCount+
        "; переходов разделов "+outline.sectionTransitions+
        ", пропусков между выбранными окнами "+
        outline.gapsBetweenSelectedWindows+
        ". Соседство окон подтверждено по смещениям; "+
        "смысловая связность разделов не подтверждена.\n"+
        "Межоконных пар для проверки "+crossPage.crossPageCandidatePairs+
        ", из разных разделов "+crossPage.crossSectionCandidatePairs+
        ", из разных этапов "+crossPage.crossStageCandidatePairs+
        ". Все ссылки пар сверены с координатами источника. "+
        "Противоречия по смыслу не подтверждены.";
    if(summary.length>2000)return null;
    return summary;
}
