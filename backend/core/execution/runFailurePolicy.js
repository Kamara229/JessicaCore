/*
 * =========================================================
 * JESSICA RUN FAILURE POLICY v4
 * =========================================================
 *
 * Анализатор результата выполнения.
 *
 *
 * Ответственность:
 *
 * Execution Result
 *        ↓
 * Failure Classification
 *        ↓
 * Normalized Failure
 *
 *
 * НЕ:
 *
 * - принимает решение retry;
 * - делает replan;
 * - вызывает Planner;
 * - вызывает Tools;
 * - создаёт финальный результат.
 *
 * =========================================================
 */





/*
 * =========================================================
 * FAILURE TYPES
 * =========================================================
 */


export const FAILURE_TYPE = {


    TOOL_ERROR:
        "tool-error",


    RUNNER_ERROR:
        "runner-error",


    SEARCH_EMPTY:
        "search-empty",


    SOURCE_UNAVAILABLE:
        "source-unavailable",


    INVALID_RESULT:
        "invalid-result",


    VALIDATION_ERROR:
        "validation-error",


    MISSING_DATA:
        "missing-data",


    USER_REQUIRED:
        "user-required",


    TEMPORARY_ERROR:
        "temporary-error",


    UNKNOWN:
        "unknown"

};









/*
 * =========================================================
 * NORMALIZE TEXT
 * =========================================================
 */


function normalizeText(
    value
) {


    return String(
        value || ""
    )
        .trim();

}









/*
 * =========================================================
 * BUILD FAILURE
 * =========================================================
 */


function buildFailure({

    stage,

    type,

    reason,

    original = null

}) {


    return {


        failed:
            true,


        stage:
            stage || "execution",



        failureType:
            type || FAILURE_TYPE.UNKNOWN,



        reason:
            reason || "Неизвестная ошибка",



        original



    };

}









/*
 * =========================================================
 * DETECT FAILURE TYPE
 * =========================================================
 */


function detectFailureType(
    result,
    reason
) {


    const text =

        reason
            .toLowerCase();





    if (
        result?.needsClarification === true
    ) {


        return FAILURE_TYPE.USER_REQUIRED;

    }







    if (
        result?.stage === "tool"
    ) {


        return FAILURE_TYPE.TOOL_ERROR;

    }








    if (
        result?.stage === "runner"
    ) {


        return FAILURE_TYPE.RUNNER_ERROR;

    }








    if (

        text.includes(
            "timeout"
        )

        ||

        text.includes(
            "network"
        )

        ||

        text.includes(
            "temporarily"
        )

    ) {


        return FAILURE_TYPE.TEMPORARY_ERROR;

    }








    if (

        text.includes(
            "not found"
        )

        ||

        text.includes(
            "empty"
        )

        ||

        text.includes(
            "нет результатов"
        )

    ) {


        return FAILURE_TYPE.SEARCH_EMPTY;

    }








    if (

        text.includes(
            "validation"
        )

        ||

        text.includes(
            "провер"
        )

    ) {


        return FAILURE_TYPE.VALIDATION_ERROR;

    }








    if (

        text.includes(
            "missing"
        )

        ||

        text.includes(
            "required"
        )

        ||

        text.includes(
            "не указан"
        )

    ) {


        return FAILURE_TYPE.MISSING_DATA;

    }







    return FAILURE_TYPE.UNKNOWN;


}









/*
 * =========================================================
 * ANALYZE RUN RESULT
 * =========================================================
 */


export function analyzeRunFailure(

    executionResult

) {


    /*
     * =====================================================
     * SUCCESS
     * =====================================================
     */


    if (
        executionResult?.success === true
    ) {


        return {


            failed:
                false,


            failureType:
                null,


            stage:
                null,


            reason:
                ""

        };


    }







    const reason =

        normalizeText(

            executionResult?.reason

            ||

            executionResult?.text

            ||

            executionResult?.error

        );







    const failureType =

        detectFailureType(

            executionResult,

            reason

        );







    return buildFailure({

        stage:

            executionResult?.stage ||

            "execution",



        type:

            executionResult?.failureType ||

            failureType,



        reason:

            reason ||

            "Не удалось выполнить операцию",



        original:

            executionResult

    });


}









/*
 * =========================================================
 * BUILD REPLANNER FEEDBACK
 * =========================================================
 *
 * Только данные для Planner.
 *
 * Не решение.
 *
 * =========================================================
 */


export function buildRunFailureFeedback(
    failure
) {


    if (
        !failure
    ) {

        return null;

    }



    return {


        previousFailure:
            {


                type:

                    failure.failureType,


                stage:

                    failure.stage,


                reason:

                    failure.reason


            }

    };

}
