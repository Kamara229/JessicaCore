/*
 * =========================================================
 * JESSICA CANDIDATE EXAMPLES v3
 * =========================================================
 *
 * Формирует компактные Execution Examples
 * для Learning Candidate.
 *
 *
 * КРИТИЧЕСКО:
 *
 * Experience Example НЕ должен хранить
 * полный Execution Result.
 *
 * Полный Result может содержать:
 *
 * - Experience Usage;
 * - Skill payload;
 * - старые Examples;
 * - вложенные diagnostics.
 *
 * Если сохранять его целиком, каждый новый
 * Improvement рекурсивно раздувает Experience.
 *
 *
 * Поэтому Example хранит только:
 *
 * task
 * compact result
 * success
 * traceId
 * executedTools
 * createdAt
 *
 *
 * Дедупликация:
 *
 * 1. traceId
 * 2. task + compact result
 *
 * =========================================================
 */


import {
    normalizeText,
    normalizeStringArray,
    mergeStringArrays,
    resolveTraceSuccess,
    resolveObservedTools
} from "./candidateUtils.js";


import {
    resolveLearningExampleSuccess
} from "../exampleSuccess.js";


const MAX_TASK_LENGTH =
    2000;


const MAX_RESULT_TEXT_LENGTH =
    2500;


/*
 * =========================================================
 * HELPERS
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


function limitText(
    value,
    maxLength
) {

    const text =
        normalizeText(
            value
        );


    if(
        text.length <= maxLength
    ){

        return text;

    }


    return (
        text.slice(
            0,
            maxLength
        )
        +
        "…"
    );

}


/*
 * =========================================================
 * COMPACT RESULT
 * =========================================================
 */


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


/*
 * =========================================================
 * CANONICAL EXAMPLE
 * =========================================================
 */


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


    const success =

        resolveLearningExampleSuccess(
            example
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

        success,

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


/*
 * =========================================================
 * TRACE EXAMPLE
 * =========================================================
 */


function buildTraceExample(
    trace
) {

    const traceId =

        trace?.id

        ||

        trace?.traceId

        ||

        null;


    const executedTools =

        resolveObservedTools(
            trace
        );


    return compactCandidateExample({

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

        executedTools,

        createdAt:

            new Date()
                .toISOString()

    });

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

        return trace.examples

            .map(
                compactCandidateExample
            )

            .filter(
                Boolean
            );

    }


    const example =

        buildTraceExample(
            trace
        );


    return example

        ? [
            example
        ]

        : [];

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
        !isObject(
            example
        )
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
            const rawExample
            of set
        ){

            const example =

                compactCandidateExample(
                    rawExample
                );


            if(
                !example
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


            result.push(
                example
            );

        }

    }


    return result;

}
