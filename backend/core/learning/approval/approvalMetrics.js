/*
 * =========================================================
 * JESSICA APPROVAL METRICS
 * =========================================================
 *
 * Канонический reader Learning Metrics.
 *
 *
 * Priority:
 *
 * proposedExperience.learning.*
 *        ↓
 * proposedExperience.*
 *
 * Верхнеуровневые поля являются
 * временным compatibility fallback.
 *
 * =========================================================
 */


function normalizeNumber(
    value
) {

    const number =
        Number(value);


    return Number.isFinite(number)

        ? number

        : null;

}


function normalizeUnit(
    value
) {

    const number =
        normalizeNumber(
            value
        );


    if(
        number === null
    ){

        return null;

    }


    return Math.max(
        0,
        Math.min(
            1,
            number
        )
    );

}


function getExperience(
    proposal
) {

    const experience =
        proposal?.proposedExperience;


    return (

        experience &&
        typeof experience === "object"

    )

        ? experience

        : {};

}


function getLearning(
    proposal
) {

    const experience =
        getExperience(
            proposal
        );


    const learning =
        experience?.learning;


    return (

        learning &&
        typeof learning === "object" &&
        !Array.isArray(learning)

    )

        ? learning

        : {};

}


function readMetric(
    proposal,
    name
) {

    const experience =
        getExperience(
            proposal
        );


    const learning =
        getLearning(
            proposal
        );


    if(
        learning[name] !== undefined &&
        learning[name] !== null
    ){

        return learning[name];

    }


    return experience[name];

}


export function getApprovalExperience(
    proposal
) {

    return getExperience(
        proposal
    );

}


export function getApprovalMetrics(
    proposal
) {

    const experience =
        getExperience(
            proposal
        );


    const examples =

        Array.isArray(
            experience.examples
        )

            ? experience.examples

            : [];


    return {

        confidence:

            normalizeUnit(

                readMetric(
                    proposal,
                    "confidence"
                )

                ??

                proposal?.confidence

            ),


        maturity:

            normalizeUnit(

                readMetric(
                    proposal,
                    "maturity"
                )

            ),


        successRate:

            normalizeUnit(

                readMetric(
                    proposal,
                    "successRate"
                )

            ),


        occurrences:

            normalizeNumber(

                readMetric(
                    proposal,
                    "occurrences"
                )

            ),


        successCount:

            normalizeNumber(

                readMetric(
                    proposal,
                    "successCount"
                )

            ),


        failureCount:

            normalizeNumber(

                readMetric(
                    proposal,
                    "failureCount"
                )

            ),


        examplesCount:

            examples.length

    };

}
