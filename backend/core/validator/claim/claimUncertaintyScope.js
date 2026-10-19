/*
 * Public facade for comparison uncertainty. Keep table parsing,
 * claim-to-answer alignment, and filtering in small focused modules.
 */
import {
    shouldDiscardUnassertedClaim,
    filterUnassertedUncertaintyClaims
} from "./uncertaintyClaimPolicy.js";

export function isUnassertedUncertaintyClaim(claim,answer,sourceUrls=[]) {
    return shouldDiscardUnassertedClaim(claim,answer,sourceUrls);
}

export function partitionComparisonClaims(claims,answer,sourceUrls) {
    return filterUnassertedUncertaintyClaims(claims,answer,sourceUrls);
}
