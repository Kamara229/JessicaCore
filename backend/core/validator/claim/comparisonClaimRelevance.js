/*
 * A conservative cross-language check for material topics in paired
 * document comparisons. It is NOT a semantic-entailment certificate:
 * a topic match cannot prove a claim, but a topic mismatch disproves
 * the proposed citation. General-purpose across source documents.
 */
const SUBJECTS=[
    {claim:/патент|patent/iu,quote:/patent|патент/iu},
    {claim:/уведомлен|авторск|копирайт|copyright|notice|attribution/iu,
     quote:/copyright|notice|attribut|retain|preserv|авторск|уведомлен/iu},
    {claim:/торгов\p{L}*\s+марк|товарн\p{L}*\s+знак|trademark|trade\s+names?/iu,
     quote:/trademark|trade\s+names?|product\s+names?|endorse|товарн|торгов/iu},
    {claim:/гаранти|ответственн|warrant|liabilit/iu,
     quote:/warrant|liabilit|as\s+is|disclaim|гаранти|ответственн/iu},
    {claim:/производн\p{L}*\s+(?:работ|произведен)|derivative\s+works?/iu,
     quote:/derivative|производн/iu},
    {claim:/прекращ\p{L}*|расторж\p{L}*|terminat/iu,
     quote:/terminat|cease|revok|прекращ|расторж/iu},
    {claim:/патентн\p{L}*\s+(?:спор|иск)|patent\s+litigation/iu,
     quote:/litigation|institute.*patent|патентн\p{L}*\s+(?:спор|иск)/iu}
];

export function unsupportedComparisonClaimTopic(claim,evidence){
    const statement=String(claim||"");
    const excerpt=String(evidence||"");
    for(const topic of SUBJECTS){
        if(topic.claim.test(statement) && !topic.quote.test(excerpt)){
            return true;
        }
    }
    return false;
}
