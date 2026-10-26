const OTHER_NEGATION=/(?<![\p{L}\p{N}_])не(?![\p{L}\p{N}_])/iu;
const EXTRA_MODALITY=/(?<![\p{L}\p{N}_])(?:может|могут|должен|должна|должны|предположительно)(?![\p{L}\p{N}_])/iu;
const BALANCED_TAIL=/^\s+и\s+не\s+применяется\s*$/iu;

/** Only identical subject and predicate scope; no free semantic substitutions. */
export function allowLegalClaimPolarityScope({
    before,after,rule,negative
}={}){
    if(typeof before!=="string"||typeof after!=="string"||
       typeof rule!=="string"||typeof negative!=="boolean")
        return false;
    if((before&&/[\p{L}\p{N}_]$/u.test(before))||
       (after&&/^[\p{L}\p{N}_]/u.test(after)))
        return false;
    // A second negation is safe only as the same literal, independent tail
    // on BOTH compared statements (included unchanged in the lookup key).
    if(OTHER_NEGATION.test(before+" "+after)&&
       !(rule==="literal:действует"&&
           !OTHER_NEGATION.test(before)&&BALANCED_TAIL.test(after)))
        return false;
    // Explicit modality must not silently be equated with plain obligations.
    if(rule!=="modal-capacity"&&
       EXTRA_MODALITY.test(before+" "+after))
        return false;
    return true;
}
