/*
 * =========================================================
 * JESSICA EXPERIENCE EXAMPLE METRICS
 * =========================================================
 */


import {
    isPositiveLearningTrace
} from "../learningOutcomePolicy.js";


import {
    resolveLearningExampleSuccess
} from "../exampleSuccess.js";


function normalizeExample(
    example
) {

    if(
        !example
        ||
        typeof example !== "object"
    ){

        return null;

    }


    return {

        ...example,

        task:

            String(
                example.task || ""
            )
                .trim(),

        success:

            resolveLearningExampleSuccess(
                example
            )

    };

}


export function getExamples(
    trace
) {

    if(
        Array.isArray(
            trace?.examples
        )
        &&
        trace.examples.length > 0
    ){

        return trace.examples

            .map(
                normalizeExample
            )

            .filter(Boolean);

    }


    return [

        {

            task:

                String(
                    trace?.task || ""
                )
                    .trim(),

            result:

                trace?.result

                ||

                null,

            success:

                isPositiveLearningTrace(
                    trace
                )

        }

    ];

}


export function calculateSuccessCount(
    examples = []
) {

    if(
        !Array.isArray(
            examples
        )
    ){

        return 0;

    }


    return examples.filter(
        item =>
            resolveLearningExampleSuccess(
                item
            )
            === true
    )
        .length;

}


export function calculateFailureCount(
    examples = []
) {

    if(
        !Array.isArray(
            examples
        )
    ){

        return 0;

    }


    return examples.filter(
        item =>
            resolveLearningExampleSuccess(
                item
            )
            === false
    )
        .length;

}


export function calculateSuccessRate(
    examples = []
) {

    if(
        !Array.isArray(
            examples
        )
        ||
        examples.length === 0
    ){

        return 0;

    }


    return Number(

        (
            calculateSuccessCount(
                examples
            )
            /
            examples.length
        )
            .toFixed(2)

    );

}
