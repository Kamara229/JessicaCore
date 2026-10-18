/*
 * =========================================================
 * JESSICA PROPOSAL ANALYSIS METRICS
 * =========================================================
 */

import {
    isObject,
    normalizeText,
    normalizeUnit,
    normalizePositiveInteger
} from "./proposalUtils.js";


function optionalNumber(
    value
) {

    if(
        value === undefined
        ||
        value === null
    ){

        return null;

    }


    const number =
        Number(value);


    return Number.isFinite(number)

        ? number

        : null;

}


export function buildProposalCurrentMetrics({

    originalAnalysis,

    candidate,

    confidence

}) {

    const originalMetrics =

        isObject(
            originalAnalysis?.metrics
        )

            ? originalAnalysis.metrics

            : {};


    const candidateOccurrences =
        optionalNumber(
            candidate?.occurrences
        );


    const candidateSuccessCount =
        optionalNumber(
            candidate?.successCount
        );


    const candidateFailureCount =
        optionalNumber(
            candidate?.failureCount
        );


    const candidateSuccessRate =
        optionalNumber(
            candidate?.successRate
        );


    const candidateMaturity =
        optionalNumber(
            candidate?.maturity
        );


    const candidateMatchScore =
        optionalNumber(
            candidate?.matchScore
        );


    const candidateConfidence =
        optionalNumber(
            candidate?.confidence
        );


    return {

        ...originalMetrics,

        occurrences:

            candidateOccurrences !== null

                ? Math.max(
                    Math.floor(
                        candidateOccurrences
                    ),
                    0
                )

                : normalizePositiveInteger(
                    originalMetrics.occurrences
                ),

        successCount:

            candidateSuccessCount !== null

                ? Math.max(
                    candidateSuccessCount,
                    0
                )

                : Math.max(
                    Number(
                        originalMetrics.successCount || 0
                    ),
                    0
                ),

        failureCount:

            candidateFailureCount !== null

                ? Math.max(
                    candidateFailureCount,
                    0
                )

                : Math.max(
                    Number(
                        originalMetrics.failureCount || 0
                    ),
                    0
                ),

        successRate:

            candidateSuccessRate !== null

                ? normalizeUnit(
                    candidateSuccessRate
                )

                : normalizeUnit(
                    originalMetrics.successRate
                ),

        maturity:

            candidateMaturity !== null

                ? normalizeUnit(
                    candidateMaturity
                )

                : normalizeUnit(
                    originalMetrics.maturity
                ),

        maturityLevel:

            normalizeText(
                candidate?.maturityLevel
            )

            ||

            normalizeText(
                originalMetrics.maturityLevel
            )

            ||

            null,

        matchScore:

            candidateMatchScore !== null

                ? normalizeUnit(
                    candidateMatchScore
                )

                : normalizeUnit(
                    originalMetrics.matchScore
                ),

        confidence:

            candidateConfidence !== null

                ? normalizeUnit(
                    candidateConfidence
                )

                : normalizeUnit(
                    confidence
                )

    };

}
