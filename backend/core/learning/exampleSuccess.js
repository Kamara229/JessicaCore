/*
 * =========================================================
 * JESSICA LEARNING EXAMPLE SUCCESS
 * =========================================================
 *
 * Каноническая оценка качества Learning Example.
 *
 * Технический success не может перекрывать
 * failed/degraded semantic validation.
 * =========================================================
 */


function normalizeStatus(
    value
) {

    return String(
        value || ""
    )
        .trim()
        .toUpperCase();

}


function normalizeOutcome(
    value
) {

    return String(
        value || ""
    )
        .trim()
        .toLowerCase();

}


function getValidation(
    example
) {

    const value =

        example
            ?.result
            ?.validation

        ||

        example?.validation

        ||

        {};


    return (

        value
        &&
        typeof value === "object"
        &&
        !Array.isArray(value)

    )

        ? value

        : {};

}


export function resolveLearningExampleSuccess(
    example
) {

    if(
        !example
        ||
        typeof example !== "object"
    ){

        return false;

    }


    const validation =

        getValidation(
            example
        );


    const outcomeType =

        normalizeOutcome(
            validation.outcomeType
        );


    if(
        outcomeType ===
        "no_verified_result"
    ){

        return false;

    }


    if(
        validation.degraded === true
        ||
        validation.valid === false
    ){

        return false;

    }


    const status =

        normalizeStatus(

            example?.status

            ||

            example
                ?.result
                ?.status

        );


    if(
        [
            "FAILED",
            "NEEDS_CLARIFICATION",
            "NO_VERIFIED_RESULT",
            "BLOCKED"
        ]
            .includes(
                status
            )
    ){

        return false;

    }


    /*
     * Strong positive evidence.
     */


    if(
        example
            ?.result
            ?.verified === true

        ||

        validation.valid === true
    ){

        return true;

    }


    /*
     * Explicit verified=false is not positive evidence
     * unless a validator explicitly said valid=true above.
     */


    if(
        example
            ?.result
            ?.verified === false
    ){

        return false;

    }


    /*
     * Legacy examples created before canonical validation
     * may not have verified/validation fields.
     *
     * Preserve them only when execution itself completed.
     */


    const explicitSuccess =

        typeof example.success === "boolean"

            ? example.success

            : (
                example
                    ?.result
                    ?.success === true
            );


    return (

        explicitSuccess === true

        &&

        (
            status === "COMPLETED"

            ||

            status === ""
        )

    );

}
