/*
 * =========================================================
 * JESSICA CANDIDATE EXAMPLE COMPACTION
 * =========================================================
 */


import {
    normalizeStringArray,
    mergeStringArrays,
    resolveTraceSuccess,
    resolveObservedTools
} from "../candidateUtils.js";


import {
    resolveLearningExampleSuccess
} from "../../exampleSuccess.js";


import {
    compactCandidateResult,
    limitCandidateTaskText
} from "./candidateResultCompact.js";


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


export function compactCandidateExample(
    example
) {

    if(
        !isObject(
            example
        )
    ){

        return null;

    }


    const traceId =

        example.traceId

        ||

        example
            ?.result
            ?.executionMeta
            ?.traceId

        ||

        null;


    const executedTools =

        mergeStringArrays(

            example.executedTools,

            example
                ?.result
                ?.executionMeta
                ?.usedTools

        );


    return {

        task:

            limitCandidateTaskText(
                example.task
            ),

        result:

            compactCandidateResult(

                example.result,

                traceId,

                executedTools

            ),

        success:

            resolveLearningExampleSuccess(
                example
            ),

        traceId,

        executedTools:

            normalizeStringArray(
                executedTools
            ),

        createdAt:

            example.createdAt

            ||

            new Date()
                .toISOString()

    };

}


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

        return trace.examples

            .map(
                compactCandidateExample
            )

            .filter(Boolean);

    }


    const traceId =

        trace?.id

        ||

        trace?.traceId

        ||

        null;


    const example =

        compactCandidateExample({

            task:
                trace?.task,

            result:

                trace?.result

                ||

                null,

            success:

                resolveTraceSuccess(
                    trace
                ),

            traceId,

            executedTools:

                resolveObservedTools(
                    trace
                ),

            createdAt:

                new Date()
                    .toISOString()

        });


    return example

        ? [
            example
        ]

        : [];

}
