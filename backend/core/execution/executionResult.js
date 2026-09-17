/*
 * =========================================================
 * JESSICA EXECUTION RESULT
 * =========================================================
 *
 * Формирует стандартизированные результаты
 * Execution Cycle.
 *
 *
 * Поддерживает:
 *
 * - COMPLETED;
 * - FAILED;
 * - NEEDS_CLARIFICATION;
 * - NO_VERIFIED_RESULT.
 *
 *
 * Этот модуль НЕ:
 *
 * - выполняет инструменты;
 * - вызывает Planner;
 * - вызывает Replanner;
 * - принимает решение о terminal outcome;
 * - вызывает Validator.
 *
 * =========================================================
 */


/*
 * =========================================================
 * USED TOOLS
 * =========================================================
 */


function collectUsedTools(
    taskRunResult
) {

    const results =
        Array.isArray(
            taskRunResult?.results
        )
            ? taskRunResult.results
            : [];


    return [

        ...new Set(

            results

                .map(
                    item =>
                        item?.tool
                )

                .filter(Boolean)

        )

    ];

}


/*
 * =========================================================
 * COMMON RESULT DATA
 * =========================================================
 */


function buildCommonData(
    context
) {

    return {

        plan:
            context?.plan || null,


        planningContext:
            context?.planningContext || {},


        toolResults:
            context?.runResult?.results || [],


        usedTools:
            collectUsedTools(
                context?.runResult
            ),


        attempt:
            Number(
                context?.attempt || 0
            )

    };

}


/*
 * =========================================================
 * COMPLETED
 * =========================================================
 */


export function buildCompletedResult(
    context,
    answerResult,
    validated
) {

    const isValidated =
        validated === true;


    return {

        success:
            true,


        status:
            "COMPLETED",


        resultType:
            "result",


        verified:
            isValidated,


        validated:
            isValidated,


        validationStatus:
            isValidated
                ? "passed"
                : "skipped",


        result:
            answerResult?.text || "",


        answerSource:
            answerResult?.source ||
            "unknown",


        ...buildCommonData(
            context
        )

    };

}


/*
 * =========================================================
 * FAILURE
 * =========================================================
 */


export function buildFailureResult(
    context,
    {
        stage = "execution",
        reason = "Не удалось выполнить задачу",
        failureType = null
    } = {}
) {

    return {

        success:
            false,


        status:
            "FAILED",


        resultType:
            "failure",


        verified:
            false,


        shouldRetry:
            false,


        stage,


        failureType,


        result:
            reason ||
            "Не удалось выполнить задачу",


        ...buildCommonData(
            context
        )

    };

}


/*
 * =========================================================
 * NEEDS CLARIFICATION
 * =========================================================
 */


export function buildClarificationResult(
    context,
    {
        stage = "execution",
        reason = "Для выполнения задачи требуется уточнение"
    } = {}
) {

    return {

        success:
            false,


        status:
            "NEEDS_CLARIFICATION",


        resultType:
            "needs_clarification",


        verified:
            false,


        shouldRetry:
            false,


        needsClarification:
            true,


        stage,


        result:
            reason ||
            "Для выполнения задачи требуется уточнение",


        ...buildCommonData(
            context
        )

    };

}


/*
 * =========================================================
 * NO VERIFIED RESULT
 * =========================================================
 *
 * Поиск или проверка были выполнены,
 * но достоверный результат подтвердить
 * не удалось.
 *
 * Это не технический FAILED.
 *
 * =========================================================
 */


export function buildNoVerifiedResult(
    context,
    {
        message =
            "Не удалось подтвердить достоверный результат по доступным источникам.",

        reason =
            "",

        stage =
            "search",

        failureType =
            null
    } = {}
) {

    return {

        success:
            true,


        status:
            "COMPLETED",


        resultType:
            "no_verified_result",


        verified:
            false,


        validated:
            false,


        validationStatus:
            "not_applicable",


        shouldRetry:
            false,


        stage,


        failureType,


        result:
            message,


        reason,


        answerSource:
            "execution",


        ...buildCommonData(
            context
        )

    };

}
