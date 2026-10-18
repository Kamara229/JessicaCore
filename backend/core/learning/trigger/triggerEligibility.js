/*
 * =========================================================
 * JESSICA LEARNING TRIGGER ELIGIBILITY
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


    // A completed child cannot justify learning from a failed parent.
    const result = trace.result;
    if (Array.isArray(result?.results) && result.results.length > 0) {
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
    } else if (result?.status === "PARTIAL" || result?.status === "FAILED") {
        return false;
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
