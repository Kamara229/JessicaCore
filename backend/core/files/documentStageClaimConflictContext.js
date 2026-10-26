const DATE=/(?<!\d)(?:0[1-9]|[12]\d|3[01])\.(?:0[1-9]|1[0-2])\.(?:19|20)\d{2}(?!\d)/gu;
const CONDITION=/(?:^|[\s,;])(?:при условии|при наличии|в случае|если)\s+([^,;.!?«»\n]{1,100})/giu;
const EXCEPTION=/(?:за исключением|кроме|исключая|если иное не предусмотрено|если только)/iu;
const MAX=8;

/** Extract visible markers only from the original cited claim and its evidence. */
export function readStageClaimContext(entry){
    const c=entry?.claim;
    if(typeof c?.claim!=="string"||typeof c?.evidence!=="string")
        return null;
    const source=(c.claim.replace(/\[О\d+:Ф1:Ф\d+\]/gu,"")+" "+
        c.evidence).replace(/\s+/gu," ").slice(0,4000);
    const dates=[...new Set([...source.matchAll(DATE)].map(x=>x[0]))].slice(0,MAX);
    const conditions=[...new Set([...source.matchAll(CONDITION)]
        .map(x=>x[0].trim().toLocaleLowerCase("ru")))].slice(0,MAX);
    return {dates,conditions,exceptionMentioned:EXCEPTION.test(source)};
}

/** Differences in wording/markers demand human review, never prove logical conflict. */
export function compareStageClaimConflictContext(pair){
    const affirmative=readStageClaimContext(pair?.affirmative);
    const negative=readStageClaimContext(pair?.negative);
    if(!affirmative||!negative)return null;
    const differences=[];
    if(affirmative.dates.length&&negative.dates.length&&
       affirmative.dates.join("|")!==negative.dates.join("|"))
        differences.push("different-explicit-dates");
    if(affirmative.conditions.length&&negative.conditions.length&&
       affirmative.conditions.join("|")!==negative.conditions.join("|"))
        differences.push("different-explicit-conditions");
    if(Boolean(affirmative.conditions.length)!==
       Boolean(negative.conditions.length))
        differences.push("condition-mentioned-on-one-side");
    if(affirmative.exceptionMentioned||negative.exceptionMentioned)
        differences.push("exception-qualification-present");
    return {affirmative,negative,differences,
        reviewStatus:differences.length?"context-qualifiers-need-review":
            "context-insufficient-to-resolve",
        semanticConflictVerified:false,fullSourceContextReviewed:false};
}
