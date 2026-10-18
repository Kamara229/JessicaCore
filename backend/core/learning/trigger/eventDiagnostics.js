/*
 * =========================================================
 * JESSICA LEARNING EVENT DIAGNOSTICS
 * =========================================================
 */


function isObject(
    value
) {

    return Boolean(

        value
        &&
        typeof value === "object"
        &&
        !Array.isArray(value)

    );

}


function normalizeText(
    value
) {

    return String(
        value || ""
    )
        .trim();

}


function normalizeNumber(
    value
) {

    const number =
        Number(value);


    return Number.isFinite(number)

        ? number

        : 0;

}


export function compactLearningAnalysis(
    analysis
) {

    if(
        !isObject(
            analysis
        )
    ){

        return null;

    }


    return {

        action:

            normalizeText(
                analysis.action
            )

            ||

            null,

        reusable:
            analysis.reusable === true,

        reason:

            normalizeText(
                analysis.reason
            ),

        analysisType:

            normalizeText(
                analysis.analysisType
            )

            ||

            null,

        improvementType:

            normalizeText(
                analysis.improvementType
            )

            ||

            null,

        discoveryRequired:
            analysis.discoveryRequired === true,

        metrics:

            isObject(
                analysis.metrics
            )

                ? {
                    ...analysis.metrics
                }

                : {},

        metadata:

            isObject(
                analysis.metadata
            )

                ? {
                    ...analysis.metadata
                }

                : {}

    };

}


export function compactLearningDecision(
    decision
) {

    if(
        !isObject(
            decision
        )
    ){

        return null;

    }


    return {

        action:

            normalizeText(
                decision.action
            )

            ||

            null,

        confidence:

            normalizeNumber(
                decision.confidence
            ),

        reusable:
            decision.reusable === true,

        reason:

            normalizeText(
                decision.reason
            ),

        createdAt:

            decision.createdAt

            ||

            null

    };

}
