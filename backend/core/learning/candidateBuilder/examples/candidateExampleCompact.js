/*
 * =========================================================
 * JESSICA CANDIDATE EXAMPLE COMPACTION
 * =========================================================
 */


import {
    normalizeText,
    normalizeStringArray,
    mergeStringArrays,
    resolveTraceSuccess,
    resolveObservedTools
} from "../candidateUtils.js";


import {
    resolveLearningExampleSuccess
} from "../../exampleSuccess.js";


const MAX_TASK_LENGTH =
    2000;


const MAX_RESULT_TEXT_LENGTH =
    2500;


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


function limitText(
    value,
    maxLength
) {

    const text =

        normalizeText(
            value
        );


    return text.length <= maxLength

        ? text

        : (
            text.slice(
                0,
                maxLength
            )
            +
            "…"
        );

}


function compactValidation(
    result
) {

    const validation =

        isObject(
            result?.validation
        )

            ? result.validation

            : {};


    return {

        valid:
            validation.valid === true,

        outcomeType:

            normalizeText(
                validation.outcomeType
            )

            ||

            null,

        validationMode:

            normalizeText(
                validation.validationMode
            )

            ||

            null,

        degraded:
            validation.degraded === true

    };

}


function compactExecutionMeta(
    result,
    fallbackTraceId,
    executedTools
) {

    const meta =

        isObject(
            result?.executionMeta
        )

            ? result.executionMeta

            : {};


    return {

        traceId:

            meta.traceId

            ||

            fallbackTraceId

            ||

            null,

        executionId:

            meta.executionId

            ||

            null,

        usedTools:

            mergeStringArrays(

                executedTools,

                meta.usedTools

            )

    };

}


function compactResult(
    result,
    fallbackTraceId,
    executedTools
) {

    const source =

        isObject(
            result
        )

            ? result

            : {};


    return {

        success:
            source.success === true,

        status:

            normalizeText(
                source.status
            )

            ||

            null,

        verified:
            source.verified === true,

        text:

            limitText(

                source?.answer?.text

                ||

                source?.text

                ||

                "",

                MAX_RESULT_TEXT_LENGTH

            )

            ||

            null,

        validation:

            compactValidation(
                source
            ),

        executionMeta:

            compactExecutionMeta(

                source,

                fallbackTraceId,

                executedTools

            )

    };

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

            limitText(
                example.task,
                MAX_TASK_LENGTH
            ),

        result:

            compactResult(

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
