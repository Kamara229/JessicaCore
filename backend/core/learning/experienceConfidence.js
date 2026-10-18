/*
 * =========================================================
 * JESSICA EXPERIENCE CONFIDENCE v4
 * =========================================================
 *
 * Расчёт evidence-метрик Experience.
 *
 *
 * ВАЖНО:
 *
 * Success для Learning
 *
 * !=
 *
 * техническое завершение Task.
 *
 *
 * Positive success определяется
 * через learningOutcomePolicy.js.
 *
 * Поэтому:
 *
 * no_verified_result
 * degraded validation
 * failed validation
 *
 * не считаются положительным
 * Experience Evidence.
 *
 * =========================================================
 */


import {
    isPositiveLearningTrace
} from "./learningOutcomePolicy.js";


import {
    resolveLearningExampleSuccess
} from "./exampleSuccess.js";


/*
 * =========================================================
 * NORMALIZE NUMBER
 * =========================================================
 */


function normalizeNumber(
    value
) {

    const number =
        Number(value);


    return Number.isFinite(number)

        ? number

        : 0;

}


/*
 * =========================================================
 * CLAMP 0..1
 * =========================================================
 */


function clampUnit(
    value
) {

    const number =

        normalizeNumber(
            value
        );


    return Math.max(

        0,

        Math.min(
            1,
            number
        )

    );

}


/*
 * =========================================================
 * POSITIVE INTEGER
 * =========================================================
 */


function normalizePositiveInteger(
    value
) {

    const number =

        Math.floor(

            normalizeNumber(
                value
            )

        );


    return number > 0

        ? number

        : 0;

}


/*
 * =========================================================
 * TRACE SUCCESS
 * =========================================================
 */


function resolveTraceSuccess(
    trace
) {

    return isPositiveLearningTrace(
        trace
    );

}


/*
 * =========================================================
 * EXAMPLE SUCCESS
 * =========================================================
 */


/*
 * =========================================================
 * NORMALIZE EXAMPLE
 * =========================================================
 */


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


/*
 * =========================================================
 * OCCURRENCES
 * =========================================================
 */


export function getOccurrences(
    trace
) {

    /*
     * Explicit Learning occurrence.
     */

    const explicit =

        normalizePositiveInteger(
            trace?.experienceOccurrences
        );


    if(
        explicit > 0
    ){

        return explicit;

    }


    /*
     * Trace statistics.
     */

    const traceOccurrences =

        normalizePositiveInteger(

            trace?.stats?.occurrences

            ??

            trace?.statistics?.occurrences

        );


    if(
        traceOccurrences > 0
    ){

        return traceOccurrences;

    }


    /*
     * Aggregated Examples.
     */

    if(
        Array.isArray(
            trace?.examples
        )

        &&

        trace.examples.length > 1
    ){

        return trace.examples.length;

    }


    /*
     * Existing Experience usage.
     */

    const usedSkills =

        Array.isArray(
            trace?.experienceUsage?.skills
        )

            ? trace.experienceUsage.skills

            : [];


    const usedSkill =

        usedSkills[0]

        ||

        null;


    if(
        usedSkill
    ){

        const storedOccurrences =

            normalizePositiveInteger(

                usedSkill
                    ?.learning
                    ?.occurrences

                ??

                usedSkill?.occurrences

            );


        if(
            storedOccurrences > 0
        ){

            return storedOccurrences + 1;

        }


        const successfulRuns =

            normalizePositiveInteger(

                usedSkill
                    ?.statistics
                    ?.successfulRuns

            );


        const failedRuns =

            normalizePositiveInteger(

                usedSkill
                    ?.statistics
                    ?.failedRuns

            );


        const runtimeOccurrences =

            successfulRuns

            +

            failedRuns;


        if(
            runtimeOccurrences > 0
        ){

            return runtimeOccurrences + 1;

        }

    }


    return 1;

}


/*
 * =========================================================
 * EXAMPLES
 * =========================================================
 */


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

            .filter(
                Boolean
            );

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

                resolveTraceSuccess(
                    trace
                )

        }

    ];

}


/*
 * =========================================================
 * SUCCESS COUNT
 * =========================================================
 */


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


/*
 * =========================================================
 * FAILURE COUNT
 * =========================================================
 */


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


/*
 * =========================================================
 * SUCCESS RATE
 * =========================================================
 */


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


    const successCount =

        calculateSuccessCount(
            examples
        );


    return Number(

        (
            successCount
            /
            examples.length
        )
            .toFixed(2)

    );

}


/*
 * =========================================================
 * CONFIDENCE
 * =========================================================
 */


export function calculateExperienceConfidence({

    matchScore = 0,

    successRate = 0,

    occurrences = 1

} = {}) {

    const normalizedMatch =

        clampUnit(
            matchScore
        );


    const normalizedSuccessRate =

        clampUnit(
            successRate
        );


    const normalizedOccurrences =

        Math.max(

            normalizePositiveInteger(
                occurrences
            ),

            1

        );


    const repeatScore =

        Math.min(

            normalizedOccurrences / 5,

            1

        );


    return Number(

        (

            normalizedMatch * 0.30

            +

            normalizedSuccessRate * 0.40

            +

            repeatScore * 0.30

        )
            .toFixed(2)

    );

}


/*
 * =========================================================
 * MATURITY
 * =========================================================
 */


export function calculateExperienceMaturity(
    occurrences
) {

    const count =

        Math.max(

            normalizePositiveInteger(
                occurrences
            ),

            1

        );


    const maturity =

        0.5

        +

        Math.min(
            count - 1,
            4
        )

        *

        0.125;


    return Number(

        Math.min(
            maturity,
            1
        )
            .toFixed(2)

    );

}


/*
 * =========================================================
 * MATURITY LEVEL
 * =========================================================
 */


export function resolveExperienceMaturityLevel(
    occurrences
) {

    const count =

        Math.max(

            normalizePositiveInteger(
                occurrences
            ),

            1

        );


    if(
        count >= 5
    ){

        return "READY";

    }


    if(
        count >= 2
    ){

        return "LEARNING";

    }


    return "NEW";

}
