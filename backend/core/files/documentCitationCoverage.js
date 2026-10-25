/** Citation traceability only: never confuse quoted fragments with reading a whole file. */
export function summarizeDocumentCitationCoverage(bundle, citedRefs = []) {
    const docs = Array.isArray(bundle?.docs) ? bundle.docs : [];
    const fragments = Array.isArray(bundle?.fragments) ? bundle.fragments : [];
    const valid = new Set(fragments.map(item => item.ref));
    const cited = new Set((Array.isArray(citedRefs) ? citedRefs : [])
        .filter(ref => typeof ref === "string" && valid.has(ref)));
    const files = docs.map((doc, index) => {
        const prefix = `Ф${index + 1}:Ф`;
        const own = fragments.filter(item => item.ref.startsWith(prefix));
        const citedCount = own.filter(item => cited.has(item.ref)).length;
        return {
            fileIndex: index,
            name: doc.name,
            availableChars: doc.text.length,
            extractionPartial: doc.truncated === true ||
                bundle?.partial === true,
            presentedFragments: own.length,
            citedFragments: citedCount,
            allPresentedFragmentsCited: own.length > 0 && citedCount === own.length
        };
    });
    return {
        completeSourceAnalysisConfirmed: false,
        extractionPartial: bundle?.partial === true,
        presentedFragments: fragments.length,
        citedFragments: cited.size,
        files
    };
}
