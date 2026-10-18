/*
 * Confidence for a new Pattern Discovery is anchored to the single
 * independently verified execution carried by its Evidence.
 * Discovery event metadata is a diagnostic snapshot, not a source
 * of extra independent examples or an authoritative success count.
 */
import { hasVerifiedPatternEvidence } from "../patternExtraction/patternEvidencePolicy.js";
import { calculateExperienceConfidence, calculateExperienceMaturity } from "../experienceConfidence.js";

export function deriveVerifiedDiscoveryMetrics(evidence) {
    if (!hasVerifiedPatternEvidence(evidence)) return null;

    const occurrences = 1;
    const successRate = 1;
    return {
        occurrences,
        successRate,
        confidence: calculateExperienceConfidence({
            matchScore: 1,
            successRate,
            occurrences
        }),
        maturity: calculateExperienceMaturity(occurrences)
    };
}
