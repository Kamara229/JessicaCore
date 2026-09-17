/*
 * =========================================================
 * JESSICA TASK RUNNER
 * STEP RESULT
 * =========================================================
 *
 * Приводит ответы инструментов
 * к единому формату TaskRunner.
 *
 *
 * Сохраняет:
 *
 * - success;
 * - data;
 * - text;
 * - reason;
 * - shouldRetry;
 * - needsClarification;
 * - stage;
 * - failureType.
 *
 * =========================================================
 */


/*
 * =========================================================
 * STRING
 * =========================================================
 */


function normalizeString(
    value
) {

    return typeof value === "string"
        ? value.trim()
        : "";

}



/*
 * =========================================================
 * NORMALIZE STEP RESULT
 * =========================================================
 */


export function normalizeStepResult(
    step,
    result
) {

    return {

        id:
            step?.id || null,


        tool:
            step?.tool || "",


        arguments:
            step?.arguments &&
            typeof step.arguments === "object"

                ? step.arguments

                : {},


        success:
            result?.success === true,


        text:
            normalizeString(
                result?.text
            ),


        reason:
            normalizeString(
                result?.reason
            ),


        data:
            result?.data ?? null,


        needsClarification:
            result?.needsClarification === true,


        shouldRetry:
            result?.shouldRetry === true,


        stage:
            normalizeString(
                result?.stage
            ),


        failureType:
            normalizeString(
                result?.failureType
            )

    };

}
