/*
 * =========================================================
 * JESSICA LEARNING TRIGGER ELIGIBILITY
 * =========================================================
 */


import {hasNegativeLearningEvidence} from "./traceEvidenceGuard.js";

function normalizeNumber(
    value
) {

    const number =
        Number(value);


    return Number.isFinite(number)

        ? number

        : 0;

}


function normalizeText(
    value
) {

    return String(
        value || ""
    )
        .trim()
        .toUpperCase();

}


export function shouldAnalyzeLearningTrace(
    trace
) {

    if(
        !trace
        ||
        typeof trace !== "object"
        ||
        Array.isArray(trace)
    ){

        return false;

    }


    // Negative verdicts always override successful task counters. Do not
    // enqueue an Experience from no_verified_result or degraded validation.
    if (hasNegativeLearningEvidence(trace)) return false;

    // A completed counter or outer status is not verified task evidence.
    // Require either verified child results or a verified terminal result.
    const result = trace.result;
    if (!result || typeof result !== "object" || Array.isArray(result)) {
        return false;
    }
    if (Array.isArray(result.results)) {
        if (result.results.length === 0) return false;
        if (result.status !== "COMPLETED" || result.success !== true) {
            return false;
        }
        if (result.results.some(child =>
            child?.status !== "COMPLETED" ||
            child?.success !== true ||
            child?.verified !== true ||
            child?.validation?.valid !== true ||
            child?.validation?.degraded === true
        )) {
            return false;
        }
    } else {
        // Terminal status alone is insufficient: success must be explicitly
        // confirmed by the independent validator.
        if (result.status !== "COMPLETED" ||
            result.success !== true ||
            result.verified !== true ||
            result.validation?.valid !== true ||
            result.validation?.degraded === true) {
            return false;
        }
    }

    const completed =

        normalizeNumber(

            trace
                ?.statistics
                ?.completed

            ??

            trace
                ?.stats
                ?.completed

        );


    if(
        completed > 0
    ){

        return true;

    }


    if(
        normalizeText(
            trace?.status
        )
        ===
        "COMPLETED"
    ){

        return true;

    }


    const resultStatus =

        normalizeText(
            trace
                ?.result
                ?.status
        );


    if(
        resultStatus ===
        "COMPLETED"
    ){

        return true;

    }


    if(
        trace
            ?.result
            ?.success === true
        &&
        ![
            "NO_VERIFIED_RESULT",
            "NEEDS_CLARIFICATION",
            "FAILED"
        ]
            .includes(
                resultStatus
            )
    ){

        return true;

    }


    return false;

}
