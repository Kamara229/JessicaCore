/*
 * =========================================================
 * JESSICA TASK RUNNER
 * RUNNER RESULT BUILDER v1
 * =========================================================
 *
 * Формирование результатов TaskRunner.
 *
 *
 * Отвечает:
 *
 * - успешный результат выполнения плана;
 * - ошибка выполнения;
 * - единый формат failure.
 *
 *
 * НЕ:
 *
 * - выполняет Tools;
 * - проверяет Plan;
 * - разрешает arguments;
 * - принимает Retry/Replan решение.
 *
 * =========================================================
 */







/*
 * =========================================================
 * BUILD FAILURE
 * =========================================================
 */


export function buildFailure({

    stage = "runner",

    failureType = "run-failure",

    text = "Не удалось выполнить план",

    reason = "",

    shouldRetry = false,

    needsClarification = false,

    failedStep = null,

    failedStepId = null,

    results = []

} = {}) {



    const finalReason =

        reason ||

        text ||

        "Не удалось выполнить план";







    return {


        success:

            false,



        shouldRetry:

            shouldRetry === true,



        needsClarification:

            needsClarification === true,



        stage,



        failureType,



        reason:

            finalReason,



        text:

            text || finalReason,



        failedStep,



        failedStepId,



        results


    };

}









/*
 * =========================================================
 * BUILD SUCCESS
 * =========================================================
 */


export function buildSuccess(

    results = []

) {


    return {


        success:

            true,



        shouldRetry:

            false,



        needsClarification:

            false,



        text:

            "План выполнен",



        results


    };

}
