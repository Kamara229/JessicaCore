/*
 * =========================================================
 * JESSICA CANDIDATE RESULT COMPACTION
 * =========================================================
 */


import {
    normalizeText,
    mergeStringArrays
} from "../candidateUtils.js";


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


export function compactCandidateResult(
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


export function limitCandidateTaskText(
    value,
    maxLength = 2000
) {

    return limitText(
        value,
        maxLength
    );

}
