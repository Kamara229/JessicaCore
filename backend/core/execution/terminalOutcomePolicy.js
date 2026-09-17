/*
 * =========================================================
 * JESSICA TERMINAL OUTCOME POLICY
 * =========================================================
 *
 * Определяет, чем должна закончиться задача,
 * если Execution Cycle больше не может делать retry.
 *
 *
 * Возможные решения:
 *
 * FAILURE
 * → произошла настоящая ошибка выполнения.
 *
 * NO_VERIFIED_RESULT
 * → Jessica корректно выполнила поиск,
 *   но не смогла подтвердить достоверный результат.
 *
 *
 * Этот модуль НЕ:
 *
 * - выполняет инструменты;
 * - вызывает Planner;
 * - вызывает Replanner;
 * - формирует пользовательский ответ через AI;
 * - изменяет PlanningContext.
 *
 * =========================================================
 */


/*
 * =========================================================
 * OUTCOME TYPES
 * =========================================================
 */


export const TERMINAL_OUTCOME = {

    FAILURE:
        "FAILURE",

    NO_VERIFIED_RESULT:
        "NO_VERIFIED_RESULT"

};


/*
 * =========================================================
 * SAFE STRING
 * =========================================================
 */


function safeString(
    value
) {

    return typeof value === "string"
        ? value.trim()
        : "";

}


/*
 * =========================================================
 * NO VERIFIED RESULT TYPES
 * =========================================================
 *
 * Эти ошибки не означают техническую поломку.
 *
 * Jessica:
 *
 * - выполнила поиск;
 * - проверила результаты;
 * - не нашла подтверждаемого источника.
 *
 * Поэтому после исчерпания попыток
 * задача может считаться корректно завершённой.
 *
 * =========================================================
 */


const NO_VERIFIED_RESULT_FAILURE_TYPES =
    new Set([

        "no-search-results",

        "no-suitable-source"

    ]);


/*
 * =========================================================
 * IS NO VERIFIED RESULT
 * =========================================================
 */


function isNoVerifiedResultFailure(
    failure
) {

    const failureType =
        safeString(
            failure?.failureType
        );


    return NO_VERIFIED_RESULT_FAILURE_TYPES
        .has(
            failureType
        );

}


/*
 * =========================================================
 * BUILD NO VERIFIED RESULT
 * =========================================================
 */


function buildNoVerifiedResult(
    failure
) {

    const reason =
        safeString(
            failure?.reason
        );


    return {

        type:
            TERMINAL_OUTCOME.NO_VERIFIED_RESULT,


        completed:
            true,


        resultType:
            "no_verified_result",


        verified:
            false,


        reason:
            reason ||
            "Не удалось подтвердить достоверный результат",


        message:
            (
                "Не удалось подтвердить достоверный результат " +
                "по доступным источникам."
            )

    };

}


/*
 * =========================================================
 * BUILD FAILURE
 * =========================================================
 */


function buildFailure(
    failure
) {

    return {

        type:
            TERMINAL_OUTCOME.FAILURE,


        completed:
            false,


        resultType:
            "failure",


        verified:
            false,


        reason:
            safeString(
                failure?.reason
            ) ||
            "Не удалось выполнить задачу"

    };

}


/*
 * =========================================================
 * RESOLVE TERMINAL OUTCOME
 * =========================================================
 */


export function resolveTerminalOutcome(
    failure
) {


    /*
     * =====================================================
     * VERIFIED RESULT NOT FOUND
     * =====================================================
     */


    if (
        isNoVerifiedResultFailure(
            failure
        )
    ) {

        return buildNoVerifiedResult(
            failure
        );

    }


    /*
     * =====================================================
     * REAL FAILURE
     * =====================================================
     */


    return buildFailure(
        failure
    );

}
