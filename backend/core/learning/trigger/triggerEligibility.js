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
