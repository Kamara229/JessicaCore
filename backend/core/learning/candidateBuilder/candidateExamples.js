/*
 * =========================================================
 * JESSICA CANDIDATE EXAMPLES
 * =========================================================
 *
 * Формирует и объединяет Execution Examples
 * для Learning Candidate.
 *
 *
 * Дедупликация:
 *
 * 1. traceId
 * 2. task + result
 *
 * =========================================================
 */


import {
    normalizeText,
    resolveTraceSuccess,
    resolveObservedTools
} from "./candidateUtils.js";


/*
 * =========================================================
 * TRACE EXAMPLE
 * =========================================================
 */


function buildTraceExample(
    trace
) {

    return {

        task:

            normalizeText(
                trace?.task
            ),


        result:

            trace?.result

            ||

            null,


        success:

            resolveTraceSuccess(
                trace
            ),


        traceId:

            trace?.id

            ||

            trace?.traceId

            ||

            null,


        /*
         * Фактически выполненные Tools.
         */


        executedTools:

            resolveObservedTools(
                trace
            ),


        createdAt:

            new Date()
                .toISOString()

    };

}


/*
 * =========================================================
 * BUILD EXAMPLES
 * =========================================================
 */


export function buildCandidateExamples(
    trace
) {

    if(
        Array.isArray(
            trace?.examples
        )
        &&
        trace.examples.length > 0
    ){

        return [

            ...trace.examples

        ];

    }


    return [

        buildTraceExample(
            trace
        )

    ];

}


/*
 * =========================================================
 * EXAMPLE KEY
 * =========================================================
 */


function buildExampleKey(
    example
) {

    if(
        !example
        ||
        typeof example !== "object"
    ){

        return "";

    }


    if(
        example.traceId
    ){

        return (

            "trace:"

            +

            String(
                example.traceId
            )

        );

    }


    const task =

        normalizeText(
            example.task
        );


    let result = "";


    try {


        result =

            JSON.stringify(
                example.result ?? null
            );


    }catch(error){


        result =

            String(
                example.result || ""
            );

    }


    return (

        task

        +

        "::"

        +

        result

    );

}


/*
 * =========================================================
 * MERGE EXAMPLES
 * =========================================================
 */


export function mergeCandidateExamples(
    ...exampleSets
) {

    const result = [];

    const known =
        new Set();


    for(
        const set
        of exampleSets
    ){

        if(
            !Array.isArray(set)
        ){

            continue;

        }


        for(
            const example
            of set
        ){

            if(
                !example
                ||
                typeof example !== "object"
            ){

                continue;

            }


            const key =

                buildExampleKey(
                    example
                );


            if(
                key
                &&
                known.has(
                    key
                )
            ){

                continue;

            }


            if(
                key
            ){

                known.add(
                    key
                );

            }


            result.push({

                ...example

            });

        }

    }


    return result;

}
