/*
 * Deterministic common-family recognition across AI-generated names
 * and categories. Only a narrowly defined procedural family is
 * recognized here. It is not a semantic approval or an evidence check.
 */
export function candidateProcedureFamily(candidate) {
    const title=[candidate?.name,
        ...(Array.isArray(candidate?.triggerPatterns)?candidate.triggerPatterns:[])].join(" ").toLowerCase();
    const workflow=(Array.isArray(candidate?.workflow)?candidate.workflow:[]).join(" ").toLowerCase();
    const comparison=/(?:compar\w*|сравн\w*|сопостав\w*|различи\w*)/i.test(title);
    const documents=/(?:document\w*|licen[sc]\w*|contract\w*|agreement\w*|polic(?:y|ies)|документ\w*|лицензи\w*)/i.test(title);
    const retrieves=/(?:fetch|retriev|obtain|source|источник|получ|загруз)/i.test(workflow);
    const criteria=/(?:criteri|table|row|compar\w*|сравн\w*|сопостав\w*|критери|таблиц)/i.test(workflow);
    return comparison && documents && retrieves && criteria
        ? "two-original-documents-comparison" : null;
}
