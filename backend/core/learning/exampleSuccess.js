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


    // An explicit failed execution or negative verification takes precedence.
    if (example?.success === false
        || example?.result?.success === false
        || example?.result?.verified === false) {
        return false;
    }

    if (example?.result?.verified === true || validation.valid === true) {
        return true;
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
