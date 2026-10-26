import {createHash} from "node:crypto";
import {extractExactFactTokens} from "./documentExactFactTokens.js";
import {supportsQuotedNumericFact} from "./documentExactFactQuoteBinding.js";
import {inspectQuotedNumericRelation} from "./documentNumericRelationAudit.js";

const digest = text => createHash("sha256").update(text, "utf8").digest("hex");
/** Audits exact literal fact tokens in ALL authenticated source windows.
 * This is NOT validation of a model's extraction, meaning or legal accuracy.
 */
export function auditDocumentExactFacts(source, verified, selected = []) {
    if (typeof source !== "string" || !source.length ||
        verified?.complete !== true || verified.allSourceWindowsVerified !== true ||
        verified.sourceChars !== source.length || digest(source) !== verified.sourceSha256 ||
        !Array.isArray(verified.windows) || verified.windows.length !== verified.totalWindows ||
        verified.windows.length > 256 || !Array.isArray(selected) || selected.length > 768)
        return null;
    let cursor = 0, found = 0, limited = false, windowsWithTokens = 0;
    const tokensByWindow = new Map();
    for (const [index, window] of verified.windows.entries()) {
        if (window?.number !== index + 1 || window.startOffset !== cursor ||
            !Number.isSafeInteger(window.endOffset) || window.endOffset <= cursor ||
            typeof window.text !== "string" ||
            source.slice(cursor, window.endOffset) !== window.text ||
            digest(window.text) !== window.textSha256)
            return null;
        const match = extractExactFactTokens(window.text, cursor);
        if (!match) return null;
        tokensByWindow.set(window.number, new Set(match.anchors.map(x=>x.value)));
        found += match.anchors.length;
        windowsWithTokens += Number(match.anchors.length > 0);
        limited ||= match.limited;
        cursor = window.endOffset;
    }
    if (cursor !== source.length) return null;
    let selectedFactTokens = 0, ungroundedSelectedFactTokens = 0;
    let unquotedSelectedFactTokens = 0;
    let numericRelationContradictions = 0;
    let numericRelationsAligned = 0;
    let numericRelationsUnresolved = 0;
    for (const claim of selected) {
        const window = verified.windows[claim?.windowNumber - 1];
        if (!window || !Number.isSafeInteger(claim.windowNumber) ||
            claim.sectionIndex !== window.sectionIndex ||
            claim.startOffset !== window.startOffset || claim.endOffset !== window.endOffset ||
            typeof claim.claim !== "string") return null;
        const statement = claim.claim.replace(/\[О[1-9]\d{0,2}:Ф1:Ф[1-9]\d{0,2}\]/gu, "");
        const matches = extractExactFactTokens(statement);
        if (!matches || matches.limited) return null;
        for (const token of matches.anchors) {
            selectedFactTokens++;
            if (!tokensByWindow.get(claim.windowNumber).has(token.value))
                ungroundedSelectedFactTokens++;
            else if (!supportsQuotedNumericFact(claim,window.text,token.value))
                unquotedSelectedFactTokens++;
            else {
                const status=inspectQuotedNumericRelation(
                    statement,claim.evidence,window.text,token.value);
                if(status==="contradiction")numericRelationContradictions++;
                else if(status==="aligned")numericRelationsAligned++;
                else numericRelationsUnresolved++;
            }
        }
    }
    return {
        exactNumericSourceWindowsAudited: verified.totalWindows,
        exactNumericSourceTokensFound: found,
        exactNumericSourceWindowsWithTokens: windowsWithTokens,
        exactNumericCandidateScanLimited: limited,
        selectedNumericTokensChecked: selectedFactTokens,
        selectedNumericTokensAbsentFromCitedWindow: ungroundedSelectedFactTokens,
        selectedNumericTokensAbsentFromQuotedEvidence: unquotedSelectedFactTokens,
        selectedNumericRelationContradictions: numericRelationContradictions,
        selectedNumericRelationsAligned: numericRelationsAligned,
        selectedNumericRelationsUnresolved: numericRelationsUnresolved,
        numericRelationsFullySemanticallyVerified: false,
        selectedNumericFactsQuoteGrounded:
            ungroundedSelectedFactTokens === 0 && unquotedSelectedFactTokens === 0,
        literalNumericTokenPresenceVerified: !limited &&
            ungroundedSelectedFactTokens === 0 && unquotedSelectedFactTokens === 0,
        factExtractionSemanticCorrectnessVerified: false,
        factExtractionCompletenessVerified: false
    };
}
