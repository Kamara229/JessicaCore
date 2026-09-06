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

import {
    analyzeRunFailure,
    buildRunFailureFeedback
} from "./runFailurePolicy.js";


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
 * ВАЖНО:
 *
 * semantic retry теперь может возникнуть
 * не только после Validator,
 * но и раньше — на уровне TaskRunner.
 *
 * Например:
 *
 * web_search
 * → Source Selector REJECT
 * → TaskRunner shouldRetry=true
 * → Replanner
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
    plan,
    taskRunResult,
    answerResult,
    attempt
) {

    return {
        success:
            true,

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
    shouldRetry = false,
    failureType = null
) {

    return {
        success:
            false,

        status:
            "FAILED",

        stage,

        shouldRetry,

        failureType,

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
 * REPLAN
 * =========================================================
 *
 * Единая функция для semantic retry.
 *
 * Её можно вызвать:
 *
 * - после ошибки маршрута TaskRunner;
 * - после отклонения Validator.
 */


async function createAlternativePlan(
    taskText,
    currentPlan,
    feedback,
    taskRunResult,
    attempt
) {

    console.log(
        "Jessica retry requested:",
        JSON.stringify({
            attempt,

            stage:
                feedback?.stage || "",

            failureType:
                feedback?.failureType || "",

            reason:
                feedback?.reason || ""
        })
    );


    let replanResult;


    try {

        replanResult =
            await replanTask(
                taskText,
                currentPlan,
                feedback,
                taskRunResult
            );

    } catch (error) {

        console.error(
            "Jessica Replanner exception:",
            error
        );


        return {
            success:
                false,

            reason:
                "Не удалось построить альтернативный план"
        };

    }


    if (
        !replanResult?.success ||
        !replanResult?.plan
    ) {

        return {
            success:
                false,

            reason:
                replanResult?.reason ||
                "Не удалось построить альтернативный план"
        };

    }


    console.log(
        "Jessica retry plan accepted:",
        JSON.stringify({
            nextAttempt:
                attempt + 1,

            intent:
                replanResult.plan.intent || ""
        })
    );


    return {
        success:
            true,

        plan:
            replanResult.plan
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


        /*
         * =================================================
         * 2. ANALYZE RUN RESULT
         * =================================================
         */


        const runFailure =
            analyzeRunFailure(
                taskRunResult
            );


        /*
         * =================================================
         * NEEDS CLARIFICATION
         * =================================================
         */


        if (
            runFailure.failed === true &&
            runFailure.needsClarification === true
        ) {

            return {
                success:
                    false,

                status:
                    "NEEDS_CLARIFICATION",

                needsClarification:
                    true,

                stage:
                    runFailure.stage ||
                    "tools",

                failureType:
                    runFailure.failureType ||
                    "needs-clarification",

                result:
                    runFailure.reason ||
                    "Для выполнения требуется уточнение",

                plan:
                    currentPlan,

                toolResults:
                    taskRunResult?.results || [],

                attempt
            };

        }


        /*
         * =================================================
         * RETRYABLE RUN FAILURE
         * =================================================
         *
         * Например:
         *
         * Source Selector отклонил всю выдачу.
         */


        if (
            runFailure.failed === true &&
            runFailure.shouldRetry === true
        ) {

            /*
             * Последняя попытка:
             * нового Replan уже не будет.
             */


            if (
                attempt >=
                MAX_EXECUTION_ATTEMPTS
            ) {

                return buildFailure(
                    runFailure.stage ||
                        "runner",

                    runFailure.reason ||
                        "Не удалось найти подходящий маршрут выполнения",

                    currentPlan,
                    taskRunResult,
                    attempt,
                    false,
                    runFailure.failureType
                );

            }


            const feedback =
                buildRunFailureFeedback(
                    runFailure
                );


            const alternative =
                await createAlternativePlan(
                    taskText,
                    currentPlan,
                    feedback,
                    taskRunResult,
                    attempt
                );


            if (
                !alternative.success
            ) {

                return buildFailure(
                    "replanner",
                    alternative.reason,
                    currentPlan,
                    taskRunResult,
                    attempt,
                    false,
                    runFailure.failureType
                );

            }


            currentPlan =
                alternative.plan;


            /*
             * Переходим к следующей итерации.
             *
             * Composer и Validator не запускаются,
             * потому что текущий route уже признан
             * непригодным.
             */


            continue;

        }


        /*
         * =================================================
         * NON-RETRYABLE RUN FAILURE
         * =================================================
         */


        if (
            runFailure.failed === true
        ) {

            return buildFailure(
                runFailure.stage ||
                    "tools",

                runFailure.reason ||
                    "Не удалось выполнить план",

                currentPlan,
                taskRunResult,
                attempt,
                false,
                runFailure.failureType
            );

        }


        /*
         * =================================================
         * 3. COMPOSE ANSWER
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
         * 4. VALIDATE
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
             * Пока сохраняем существующую политику:
             *
             * если сам Validator технически сломался,
             * уже сформированный ответ не выбрасываем.
             */


            return {
                success:
                    true,

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
         * 5. VALIDATOR NEEDS CLARIFICATION
         * =================================================
         */


        if (
            validation?.needsClarification === true
        ) {

            return {
                success:
                    false,

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
         * 6. VALID
         * =================================================
         */


        if (
            validation?.valid === true
        ) {

            return buildCompletedResult(
                currentPlan,
                taskRunResult,
                answerResult,
                attempt
            );

        }


        /*
         * =================================================
         * 7. VALIDATOR RETRY POLICY
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
                false,
                "validation-failure"
            );

        }


        /*
         * =================================================
         * 8. REPLAN AFTER VALIDATOR
         * =================================================
         */


        const alternative =
            await createAlternativePlan(
                taskText,
                currentPlan,
                validation,
                taskRunResult,
                attempt
            );


        if (
            !alternative.success
        ) {

            return buildFailure(
                "replanner",
                alternative.reason,
                currentPlan,
                taskRunResult,
                attempt
            );

        }


        currentPlan =
            alternative.plan;

    }


    /*
     * =====================================================
     * SAFETY FALLBACK
     * =====================================================
     */


    return buildFailure(
        "execution",

        previousValidation?.reason ||
            previousRunResult?.text ||
            "Исчерпан лимит попыток выполнения",

        currentPlan,
        previousRunResult,
        MAX_EXECUTION_ATTEMPTS
    );

}
