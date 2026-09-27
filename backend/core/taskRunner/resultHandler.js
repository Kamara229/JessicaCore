/*
 * =========================================================
 * JESSICA TASK RUNNER
 * RESULT HANDLER v1
 * =========================================================
 *
 * Формирование результатов TaskRunner.
 *
 *
 * Отвечает:
 *
 * - единый формат ошибок;
 * - единый формат успеха;
 * - сохранение failed step информации.
 *
 *
 * НЕ:
 *
 * - выполняет шаги;
 * - вызывает Tools;
 * - делает Retry;
 * - делает Replan.
 *
 * =========================================================
 */







/*
 * =========================================================
 * FAILURE RESULT
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

}) {


    const finalReason =

        reason ||

        text ||

        "Не удалось выполнить план";




    return {


        success:false,



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
 * SUCCESS RESULT
 * =========================================================
 */


export function buildSuccess(

    results = []

) {


    return {


        success:true,


        shouldRetry:false,


        needsClarification:false,


        text:
            "План выполнен",



        results


    };

}









/*
 * =========================================================
 * EXCEPTION RESULT
 * =========================================================
 */


export function buildExceptionFailure({

    stage,

    error,

    failedStep = null,

    failedStepId = null,

    results = []

}) {


    return buildFailure({

        stage,


        failureType:

            `${stage}-exception`,



        text:

            error?.message ||

            `Ошибка этапа ${stage}`,



        failedStep,


        failedStepId,


        results


    });

}









/*
 * =========================================================
 * INVALID PLAN
 * =========================================================
 */


export function buildInvalidPlanFailure(

    validation

) {


    return buildFailure({

        stage:

            validation?.stage ||

            "runner",



        failureType:

            validation?.failureType ||

            "invalid-plan",



        text:

            validation?.text ||

            "Некорректный план",



        shouldRetry:

            validation?.shouldRetry === true


    });

}









/*
 * =========================================================
 * INVALID STEP
 * =========================================================
 */


export function buildInvalidStepFailure(

    index,

    results = []

) {


    return buildFailure({

        stage:
            "runner",



        failureType:
            "invalid-step",



        text:

            `Некорректный шаг ${index + 1}`,



        failedStep:
            index,



        results


    });

}









/*
 * =========================================================
 * MISSING TOOL
 * =========================================================
 */


export function buildMissingToolFailure({

    index,

    stepId,

    results = []

}) {


    return buildFailure({

        stage:
            "runner",



        failureType:
            "missing-tool",



        text:

            `В шаге ${index + 1} отсутствует tool`,



        failedStep:
            index,



        failedStepId:
            stepId,



        results


    });

}









/*
 * =========================================================
 * UNKNOWN TOOL
 * =========================================================
 */


export function buildUnknownToolFailure({

    toolName,

    index,

    stepId,

    results = []

}) {


    return buildFailure({

        stage:
            "runner",



        failureType:
            "unknown-tool",



        text:

            `Инструмент ${toolName} не зарегистрирован`,



        failedStep:
            index,



        failedStepId:
            stepId,



        results


    });

}
