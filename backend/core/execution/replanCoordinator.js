import {
    replanTask
} from "../replanner.js";


/*
 * =========================================================
 * JESSICA REPLAN COORDINATOR
 * =========================================================
 *
 * Координирует переход от неудачной попытки
 * выполнения к новому плану.
 *
 *
 * Flow:
 *
 * Failure / Validation Failure
 *        ↓
 * Replan Feedback
 *        ↓
 * replanTask()
 *        ↓
 * Alternative Plan
 *        ↓
 * Apply To Execution Context
 *
 *
 * Этот модуль НЕ:
 *
 * - строит Planner prompt;
 * - вызывает AI напрямую;
 * - анализирует Retry History;
 * - выполняет инструменты;
 * - запускает Execution Cycle;
 * - принимает terminal outcome.
 *
 *
 * Retry History и Retry Context
 * находятся внутри Replanner.
 *
 * =========================================================
 */


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
 * VALIDATOR FEEDBACK
 * =========================================================
 *
 * Приводит результат Validator
 * к общему failure-формату Replanner.
 *
 * =========================================================
 */


export function buildValidatorFeedback(
    validation
) {

    return {

        stage:
            "validator",


        failureType:
            "validation-failure",


        reason:
            safeString(
                validation?.reason
            ) ||
            "Ответ не прошёл проверку",


        shouldRetry:
            true,


        needsClarification:
            false

    };

}


/*
 * =========================================================
 * NORMALIZE FEEDBACK
 * =========================================================
 *
 * Replanner должен всегда получать
 * одинаковую структуру ошибки.
 *
 * =========================================================
 */


function normalizeReplanFeedback(
    feedback
) {

    return {

        stage:
            safeString(
                feedback?.stage
            ) ||
            "execution",


        failureType:
            safeString(
                feedback?.failureType
            ) ||
            "execution-failure",


        reason:
            safeString(
                feedback?.reason ||
                feedback?.text
            ) ||
            "Предыдущий маршрут выполнения оказался неэффективным",


        shouldRetry:
            feedback?.shouldRetry !== false,


        needsClarification:
            feedback?.needsClarification === true

    };

}


/*
 * =========================================================
 * CREATE ALTERNATIVE PLAN
 * =========================================================
 */


export async function createAlternativePlan(
    context,
    feedback
) {

    const normalizedFeedback =
        normalizeReplanFeedback(
            feedback
        );


    console.log(
        "Jessica Replan requested:",
        JSON.stringify({

            attempt:
                context?.attempt || 0,

            stage:
                normalizedFeedback.stage,

            failureType:
                normalizedFeedback.failureType,

            reason:
                normalizedFeedback.reason

        })
    );


    /*
     * Replanner не должен запускаться,
     * если уже известно, что требуется
     * уточнение пользователя.
     */


    if (
        normalizedFeedback.needsClarification === true
    ) {

        return {

            success:
                false,

            reason:
                normalizedFeedback.reason,

            needsClarification:
                true

        };

    }


    try {

        const result =
            await replanTask(

                context?.task || "",

                context?.plan || null,

                normalizedFeedback,

                context?.runResult || null,

                context?.planningContext || {}

            );


        /*
         * =================================================
         * REPLANNER FAILURE
         * =================================================
         */


        if (
            !result?.success ||
            !result?.plan
        ) {

            return {

                success:
                    false,

                reason:
                    result?.reason ||
                    "Replanner не создал новый план"

            };

        }


        /*
         * =================================================
         * SUCCESS
         * =================================================
         */


        return {

            success:
                true,


            plan:
                result.plan,


            planningContext:

                result?.context &&
                typeof result.context === "object" &&
                !Array.isArray(
                    result.context
                )

                    ? result.context

                    : context?.planningContext || {}

        };

    } catch (error) {

        console.error(
            "Jessica Replan Coordinator error:",
            error
        );


        return {

            success:
                false,

            reason:
                error?.message ||
                "Ошибка создания альтернативного плана"

        };

    }

}


/*
 * =========================================================
 * APPLY ALTERNATIVE PLAN
 * =========================================================
 *
 * Обновляет Execution Context после успешного Replan.
 *
 *
 * Важно:
 *
 * старые answerResult/runResult относятся
 * к предыдущему маршруту.
 *
 * Поэтому перед следующей попыткой
 * очищаем их.
 *
 * =========================================================
 */


export function applyAlternativePlan(
    context,
    alternative
) {

    if (
        !context ||
        typeof context !== "object"
    ) {

        return false;

    }


    if (
        !alternative?.success ||
        !alternative?.plan
    ) {

        return false;

    }


    context.plan =
        alternative.plan;


    context.planningContext =
        alternative.planningContext ||
        context.planningContext ||
        {};


    context.runResult =
        null;


    context.answerResult =
        null;


    return true;

}
