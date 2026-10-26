import {extractExactFactTokens} from "./documentExactFactTokens.js";

const QUOTE=/«([^»]{1,1200})»/gu;
const compact=text=>text.replace(/\s+/gu," ").trim();

/** Verify the claimed literal number is inside a source-located quote.
 * Matching tokens does not establish the factual meaning of the statement.
 */
export function supportsQuotedNumericFact(claim, sourceWindow, value) {
    if(typeof claim?.evidence!=="string"||
       typeof sourceWindow!=="string"||typeof value!=="string")return false;
    const normalizedSource=compact(sourceWindow);
    for(const match of claim.evidence.matchAll(QUOTE)){
        const quote=compact(match[1]);
        if(!quote||!normalizedSource.includes(quote))continue;
        const tokens=extractExactFactTokens(quote);
        if(tokens&&!tokens.limited&&
           tokens.anchors.some(anchor=>compact(anchor.value)===compact(value)))
            return true;
    }
    return false;
}
