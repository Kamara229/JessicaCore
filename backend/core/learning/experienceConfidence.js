/*
 * =========================================================
 * JESSICA EXPERIENCE CONFIDENCE FACADE v5
 * =========================================================
 *
 * Тонкий facade над:
 *
 * - occurrence counting;
 * - example quality;
 * - success metrics;
 * - confidence / maturity.
 * =========================================================
 */


export {
    getOccurrences
} from "./confidence/experienceOccurrences.js";


export {
    getExamples,
    calculateSuccessCount,
    calculateFailureCount,
    calculateSuccessRate
} from "./confidence/experienceExamples.js";


export {
    calculateExperienceConfidence,
    calculateExperienceMaturity,
    resolveExperienceMaturityLevel
} from "./confidence/experienceMetrics.js";
