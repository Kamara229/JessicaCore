/*
 * =========================================================
 * JESSICA PATTERN EVIDENCE v1
 * =========================================================
 *
 * Формирует Evidence для автономного
 * Experience Pattern Extraction.
 *
 *
 * Execution Trace
 *        ↓
 * Pattern Evidence
 *
 *
 * Ответственность:
 *
 * - извлечь Task;
 * - извлечь Answer;
 * - извлечь Validation;
 * - извлечь Tools;
 * - извлечь Steps;
 * - извлечь Failures;
 * - добавить Learning Metrics.
 *
 *
 * НЕ:
 *
 * - вызывает AI;
 * - создаёт Pattern;
 * - принимает Learning Decision.
 *
 * =========================================================
 */





function normalizeText(
    value,
    maxLength = 4000
) {

    return String(
        value || ""
    )
    .trim()
    .slice(
        0,
        maxLength
    );

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





function normalizeUnit(
    value
) {

    return Math.max(

        0,

        Math.min(

            1,

            normalizeNumber(
                value
            )

        )

    );

}





/*
 * =========================================================
 * ANSWER
 * =========================================================
 */


function extractAnswerText(
    trace
) {

    const values = [

        trace?.result?.answer?.text,

        trace?.result?.text,

        trace?.answer?.text,

        trace?.answer,

        trace?.resultText

    ];



    for(
        const value
        of values
    ){

        const text =

            normalizeText(
                value,
                3000
            );


        if(
            text
        ){

            return text;

        }

    }



    return "";

}





/*
 * =========================================================
 * TOOLS
 * =========================================================
 */


function extractUsedTools(
    trace
) {

    const tools = [];



    const addTool = (
        value
    ) => {


        const name =

            normalizeText(
                value,
                150
            );


        if(
            !name
        ){

            return;

        }


        if(
            !tools.includes(
                name
            )
        ){

            tools.push(
                name
            );

        }

    };



    const directTools = [

        ...(Array.isArray(trace?.usedTools)
            ? trace.usedTools
            : []),

        ...(Array.isArray(trace?.tools)
            ? trace.tools
            : [])

    ];



    for(
        const item
        of directTools
    ){

        addTool(

            typeof item === "string"

                ? item

                : (

                    item?.name

                    ||

                    item?.tool

                    ||

                    item?.toolName

                )

        );

    }



    const steps =

        Array.isArray(
            trace?.steps
        )

            ? trace.steps

            : [];



    for(
        const step
        of steps
    ){

        addTool(

            step?.tool

            ||

            step?.toolName

            ||

            step?.metadata?.tool

        );

    }



    return tools.slice(
        0,
        20
    );

}





/*
 * =========================================================
 * STEPS
 * =========================================================
 */


function extractSteps(
    trace
) {

    const steps =

        Array.isArray(
            trace?.steps
        )

            ? trace.steps

            : [];



    return steps

        .slice(
            -20
        )

        .map(

            step => ({

                stage:

                    normalizeText(
                        step?.stage ||
                        step?.name,
                        120
                    ),


                status:

                    normalizeText(
                        step?.status,
                        80
                    ),


                tool:

                    normalizeText(
                        step?.tool ||
                        step?.toolName ||
                        step?.metadata?.tool,
                        120
                    ),


                reason:

                    normalizeText(
                        step?.reason,
                        500
                    )

            })

        );

}





/*
 * =========================================================
 * FAILURES
 * =========================================================
 */


function extractFailures(
    trace
) {

    const failures =

        Array.isArray(
            trace?.failures
        )

            ? trace.failures

            : [];



    return failures

        .slice(
            -10
        )

        .map(

            failure => ({

                stage:

                    normalizeText(
                        failure?.stage,
                        120
                    ),


                failureType:

                    normalizeText(
                        failure?.failureType,
                        150
                    ),


                category:

                    normalizeText(
                        failure?.category,
                        120
                    ),


                reason:

                    normalizeText(
                        failure?.reason,
                        700
                    )

            })

        );

}





/*
 * =========================================================
 * BUILD EVIDENCE
 * =========================================================
 */


export function buildPatternEvidence({

    trace,

    metrics = {}

} = {}) {


    return {


        task:

            normalizeText(
                trace?.task,
                3000
            ),



        answer:

            extractAnswerText(
                trace
            ),



        result: {


            success:

                trace?.result?.success === true

                ||

                trace?.success === true,


            status:

                normalizeText(

                    trace?.result?.status

                    ||

                    trace?.status,

                    100

                ),


            terminalType:

                normalizeText(

                    trace
                        ?.result
                        ?.terminal
                        ?.type

                    ||

                    trace
                        ?.terminal
                        ?.type,

                    100

                )

        },



        validation: {


            valid:

                trace
                    ?.result
                    ?.validation
                    ?.valid === true

                ||

                trace
                    ?.validation
                    ?.valid === true,


            outcomeType:

                normalizeText(

                    trace
                        ?.result
                        ?.validation
                        ?.outcomeType

                    ||

                    trace
                        ?.validation
                        ?.outcomeType,

                    120

                ),


            reason:

                normalizeText(

                    trace
                        ?.result
                        ?.validation
                        ?.reason

                    ||

                    trace
                        ?.validation
                        ?.reason,

                    1000

                )

        },



        tools:

            extractUsedTools(
                trace
            ),



        steps:

            extractSteps(
                trace
            ),



        failures:

            extractFailures(
                trace
            ),



        replans:

            Array.isArray(
                trace?.replans
            )

                ? trace.replans.length

                : normalizeNumber(
                    trace?.statistics?.replans
                ),



        metrics: {


            occurrences:

                Math.max(

                    normalizeNumber(
                        metrics?.occurrences
                    ),

                    1

                ),


            successRate:

                normalizeUnit(
                    metrics?.successRate
                ),


            maturity:

                normalizeUnit(
                    metrics?.maturity
                )

        }

    };

}
