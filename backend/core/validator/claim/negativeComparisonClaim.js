/*
 * A quote may exist verbatim and still fail to support the claim.
 * In particular, "No additional restrictions" says nothing about
 * whether ShareAlike is required. Fail closed on comparative claims
 * about the ABSENCE of a provision unless evidence explicitly denies
 * that very requirement.
 */
function isNegativeRequirement(text) {
    return /(?:не\s+(?:требует(?:ся)?|предусматрива(?:ет|ется)|содержит)|нет\s+(?:услови|обязательств|требован)|отсутств(?:ует|уют|ие|ия)|does\s+not\s+(?:require|contain|include)|not\s+required|no\s+requirement|without\s+(?:a\s+)?requirement)/iu
        .test(String(text || ""));
}

function explicitlyNegatesRequirement(text) {
    return /(?:not\s+required|no\s+(?:requirement|obligation)|does\s+not\s+(?:require|contain|include)|not\s+obliged|не\s+требует(?:ся)?|требовани\p{L}*\s+отсутств\p{L}*)/iu
        .test(String(text || ""));
}

function distinctiveTerms(claim) {
    // Retain explicit capitalized named provisions without
    // relying on language-specific translations.
    const words=String(claim || "").match(/\b[A-Z][A-Za-z]{4,}\b/g) || [];
    return words.map(x=>x.toLowerCase()).filter(x=>
        !["license","source","clause","copyright","document"].includes(x));
}

export function isUnsupportedNegativeComparisonClaim(claim, multiSource = false) {
    if(!multiSource || !isNegativeRequirement(claim?.claim)) return false;
    const evidence=String(claim?.evidence || "");
    if(!explicitlyNegatesRequirement(evidence)) return true;
    const terms=distinctiveTerms(claim.claim);
    // "Not required" about an unrelated provision is not evidence.
    if(terms.length && !terms.some(t=>evidence.toLowerCase().includes(t))) return true;
    return false;
}
