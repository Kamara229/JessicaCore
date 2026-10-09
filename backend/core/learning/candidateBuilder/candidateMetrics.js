/*
 * =========================================================
 * JESSICA CANDIDATE METRICS
 * =========================================================
 *
 * Формирует Learning Metrics
 * для Experience Candidate.
 *
 * =========================================================
 */


import {
    calculateSuccessCount,
    calculateFailureCount,
    calculateSuccessRate,
    resolveExperienceMaturityLevel
} from "../experienceConfidence.js";


import {
    normalizeNumber,
    normalizeUnit
} from "./candidateUtils.js";


/*
 * =========================================================
 * BUILD METRICS
 * =========================================================
 */


export function buildCandidateLearningMetrics({

    examples,

    occurrences,

    maturity,

    confidence

}) {

    const normalizedExamples =

        Array.isArray(
            examples
        )

            ? examples

            : [];


    const successCount =

        calculateSuccessCount(
            normalizedExamples
        );


    const failureCount =

        calculateFailureCount(
            normalizedExamples
        );


    const successRate =

        calculateSuccessRate(
            normalizedExamples
        );


    const normalizedOccurrences =

        Math.max(

            Math.floor(

                normalizeNumber(
                    occurrences
                )

            ),

            normalizedExamples.length,

            1

        );


    return {

        occurrences:

            normalizedOccurrences,


        successCount,


        failureCount,


        successRate,


        maturity:

            normalizeUnit(
                maturity
            ),


        maturityLevel:

            resolveExperienceMaturityLevel(
                normalizedOccurrences
            ),


        confidence:

            normalizeUnit(
                confidence
            )

    };

}
