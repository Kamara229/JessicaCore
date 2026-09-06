import {
    runPlan
} from "../taskRunner.js";

import {
    composeAnswer
} from "../answerComposer.js";

import {
    validateResult
} from "../validator.js";

import {
    replanTask
} from "../replanner.js";

import {
    MAX_EXECUTION_ATTEMPTS,
    shouldRetryExecution
} from "./retryPolicy.js";


/*
 * =========================================================
 * JESSICA EXECUTION CYCLE
 * =========================================================
 *
 * Выполняет цикл одной подзадачи:
 *
 * plan
 * → run
 * → compose
 * → validate
 * → replan
 * → retry
 *
 * Сам первый план не создаёт.
 * Его передаёт subtaskRunner.
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


    return results
        .map(
            item =>
                item?.tool
        )
        .filter(Boolean);
}


/*
 * =========================================================
 * COMPLETED RESULT
 * =========================================================
 */


function buildCompletedResult(
    taskText,
    plan,
    taskRunResult,
    answerResult,
    attempt
) {

    return {
        success: true,

        status:
            "COMPLETED",

        validated:
            true,

        result:
            answerResult.text,

        answerSource:
            answerResult.source || "unknown",

        usedTools:
            collectUsedTools(
                taskRunResult
            ),

        plan,

        toolResults:
            taskRunResult?.results || [],

        attempt
    };
}


/*
 * =========================================================
 * EXECUTION FAILURE
 * =========================================================
 */


function buildFailure(
    stage,
    text,
    plan,
    taskRunResult,
    attempt,
    shouldRetry = false
) {

    return {
        success: false,

        status:
            "FAILED",

        stage,

        shouldRetry,

        result:
            text,

        plan,

        toolResults:
            taskRunResult?.results || [],

        attempt
    };
}


/*
 * =========================================================
 * EXECUTE CYCLE
 * =========================================================
 */


export async function executePlanCycle(
    taskText,
    initialPlan
) {

    let currentPlan =
        initialPlan;


    let previousRunResult =
        null;


    let previousValidation =
        null;


    for (
        let attempt = 1;
        attempt <= MAX_EXECUTION_ATTEMPTS;
        attempt++
    ) {

        console.log(
            `Jessica execution attempt ${attempt}/${MAX_EXECUTION_ATTEMPTS}`
        );


        /*
         * =================================================
         * 1. RUN PLAN
         * =================================================
         */


        let taskRunResult;


        try {

            taskRunResult =
                await runPlan(
                    currentPlan,
                    taskText
                );

        } catch (error) {

            console.error(
                "Jessica execution runner error:",
                error
            );


            return buildFailure(
                "runner",
                "Ошибка выполнения плана",
                currentPlan,
                previousRunResult,
                attempt
            );

        }


        previousRunResult =
            taskRunResult;


        if (
            taskRunResult?.needsClarification === true
        ) {

            return {
                success: false,

                status:
                    "NEEDS_CLARIFICATION",

                needsClarification:
                    true,

                stage:
                    "tools",

                result:
                    taskRunResult.text ||
                    "Для выполнения требуется уточнение",

                plan:
                    currentPlan,

                toolResults:
                    taskRunResult.results || [],

                attempt
            };

        }


        if (
            taskRunResult?.success !== true
        ) {

            return buildFailure(
                "tools",
                taskRunResult?.text ||
                    "Не удалось выполнить план",
                currentPlan,
                taskRunResult,
                attempt
            );

        }


        /*
         * =================================================
         * 2. COMPOSE ANSWER
         * =================================================
         */


        let answerResult;


        try {

            answerResult =
                await composeAnswer(
                    taskText,
                    currentPlan,
                    taskRunResult
                );

        } catch (error) {

            console.error(
                "Jessica execution composer error:",
                error
            );


            return buildFailure(
                "composer",
                "Не удалось сформировать ответ",
                currentPlan,
                taskRunResult,
                attempt
            );

        }


        if (
            !answerResult?.success
        ) {

            return buildFailure(
                "composer",
                answerResult?.text ||
                    "Не удалось сформировать ответ",
                currentPlan,
                taskRunResult,
                attempt
            );

        }


        /*
         * =================================================
         * 3. VALIDATE
         * =================================================
         */


        let validation;


        try {

            validation =
                await validateResult(
                    taskText,
                    currentPlan,
                    taskRunResult,
                    answerResult
                );

        } catch (error) {

            console.error(
                "Jessica execution validator error:",
                error
            );


            /*
             * Пока сохраняем старое поведение:
             * если Validator технически сломался,
             * уже полученный ответ не выбрасываем.
             */
            return {
                success: true,

                status:
                    "COMPLETED",

                validated:
                    false,

                result:
                    answerResult.text,

                answerSource:
                    answerResult.source || "unknown",

                usedTools:
                    collectUsedTools(
                        taskRunResult
                    ),

                plan:
                    currentPlan,

                toolResults:
                    taskRunResult.results || [],

                attempt
            };

        }


        previousValidation =
            validation;


        /*
         * =================================================
         * 4. NEEDS CLARIFICATION
         * =================================================
         */


        if (
            validation?.needsClarification === true
        ) {

            return {
                success: false,

                status:
                    "NEEDS_CLARIFICATION",

                needsClarification:
                    true,

                stage:
                    "validator",

                result:
                    validation.reason ||
                    "Для выполнения требуется уточнение",

                plan:
                    currentPlan,

                toolResults:
                    taskRunResult.results || [],

                attempt
            };

        }


        /*
         * =================================================
         * 5. VALID
         * =================================================
         */


        if (
            validation?.valid === true
        ) {

            return buildCompletedResult(
                taskText,
                currentPlan,
                taskRunResult,
                answerResult,
                attempt
            );

        }


        /*
         * =================================================
         * 6. RETRY POLICY
         * =================================================
         */


        if (
            !shouldRetryExecution(
                validation,
                attempt
            )
        ) {

            return buildFailure(
                "validator",
                validation?.reason ||
                    "Результат не прошёл проверку качества",
                currentPlan,
                taskRunResult,
                attempt,
                validation?.shouldRetry === true
            );

        }


        /*
         * =================================================
         * 7. REPLAN
         * =================================================
         */


        console.log(
            "Jessica retry requested:",
            JSON.stringify({
                attempt,
                reason:
                    validation?.reason || ""
            })
        );


        let replanResult;


        try {

            replanResult =
                await replanTask(
                    taskText,
                    currentPlan,
                    validation,
                    taskRunResult
                );

        } catch (error) {

            console.error(
                "Jessica Replanner exception:",
                error
            );


            return buildFailure(
                "replanner",
                "Не удалось построить альтернативный план",
                currentPlan,
                taskRunResult,
                attempt
            );

        }


        if (
            !replanResult?.success ||
            !replanResult?.plan
        ) {

            return buildFailure(
                "replanner",
                replanResult?.reason ||
                    "Не удалось построить альтернативный план",
                currentPlan,
                taskRunResult,
                attempt
            );

        }


        /*
         * Следующая итерация
         * выполняет уже новый план.
         */
        currentPlan =
            replanResult.plan;


        console.log(
            "Jessica retry plan accepted:",
            JSON.stringify({
                nextAttempt:
                    attempt + 1,

                intent:
                    currentPlan.intent || ""
            })
        );

    }


    /*
     * Теоретически сюда не должны попасть,
     * но оставляем безопасный fallback.
     */
    return buildFailure(
        "execution",
        previousValidation?.reason ||
            "Исчерпан лимит попыток выполнения",
        currentPlan,
        previousRunResult,
        MAX_EXECUTION_ATTEMPTS
    );
}
