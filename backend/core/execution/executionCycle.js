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
    MAX_EXECUTION_ATTEMPTS,
    shouldRetryExecution
} from "./retryPolicy.js";

import {
    analyzeRunFailure,
    buildRunFailureFeedback
} from "./runFailurePolicy.js";

import {
    TERMINAL_OUTCOME,
    resolveTerminalOutcome
} from "./terminalOutcomePolicy.js";

import {
    buildCompletedResult,
    buildFailureResult,
    buildClarificationResult,
    buildNoVerifiedResult
} from "./executionResult.js";

import {
    buildValidatorFeedback,
    createAlternativePlan,
    applyAlternativePlan
} from "./replanCoordinator.js";


/*
 * =========================================================
 * JESSICA EXECUTION CYCLE
 * =========================================================
 *
 * Центральный координатор выполнения плана.
 *
 *
 * Flow:
 *
 * Plan
 *   ↓
 * TaskRunner
 *   ↓
 * Run Failure Analysis
 *   ↓
 * Answer Composer
 *   ↓
 * Validator
 *   ↓
 * Replanner
 *   ↓
 * Retry
 *
 *
 * После исчерпания допустимых попыток:
 *
 * Terminal Outcome Policy
 *
 *
 * Этот файл НЕ:
 *
 * - строит результаты вручную;
 * - вызывает Replanner напрямую;
 * - хранит Retry History;
 * - анализирует terminal outcome самостоятельно;
 * - ищет Experience;
 * - строит Planner prompt.
 *
 * =========================================================
 */


/*
 * =========================================================
 * TERMINAL RESULT
 * =========================================================
 */


function buildTerminalResult(
    context,
    failure
) {

    const outcome =
        resolveTerminalOutcome(
            failure
        );


    console.log(
        "Jessica terminal outcome:",
        JSON.stringify({

            type:
                outcome?.type || null,

            resultType:
                outcome?.resultType || null,

            failureType:
                failure?.failureType || null,

            attempt:
                context?.attempt || 0

        })
    );


    /*
     * =====================================================
     * NO VERIFIED RESULT
     * =====================================================
     */


    if (
        outcome?.type ===
        TERMINAL_OUTCOME.NO_VERIFIED_RESULT
    ) {

        return buildNoVerifiedResult(

            context,

            {

                message:
                    outcome?.message,

                reason:
                    outcome?.reason ||
                    failure?.reason ||
                    "",

                stage:
                    failure?.stage ||
                    "search",

                failureType:
                    failure?.failureType ||
                    null

            }

        );

    }


    /*
     * =====================================================
     * REAL FAILURE
     * =====================================================
     */


    return buildFailureResult(

        context,

        {

            stage:
                failure?.stage ||
                "execution",

            reason:
                outcome?.reason ||
                failure?.reason ||
                "Не удалось выполнить задачу",

            failureType:
                failure?.failureType ||
                null

        }

    );

}


/*
 * =========================================================
 * APPLY REPLAN
 * =========================================================
 */


async function tryReplan(
    context,
    feedback
) {

    const alternative =
        await createAlternativePlan(

            context,

            feedback

        );


    if (
        !alternative?.success
    ) {

        return {

            success:
                false,

            result:
                buildFailureResult(

                    context,

                    {

                        stage:
                            "replanner",

                        reason:
                            alternative?.reason ||
                            "Replanner не создал новый план",

                        failureType:
                            "replanner-failure"

                    }

                )

        };

    }


    const applied =
        applyAlternativePlan(

            context,

            alternative

        );


    if (
        applied !== true
    ) {

        return {

            success:
                false,

            result:
                buildFailureResult(

                    context,

                    {

                        stage:
                            "replanner",

                        reason:
                            "Не удалось применить альтернативный план",

                        failureType:
                            "replan-apply-failure"

                    }

                )

        };

    }


    return {

        success:
            true

    };

}


/*
 * =========================================================
 * MAIN EXECUTION
 * =========================================================
 */


export async function executePlanCycle(

    taskText,

    initialPlan,

    planningContext = {}

) {


    /*
     * =====================================================
     * EXECUTION CONTEXT
     * =====================================================
     */


    const context = {

        task:
            taskText,

        plan:
            initialPlan,

        planningContext,

        runResult:
            null,

        answerResult:
            null,

        attempt:
            0

    };


    let lastFailure =
        null;


    /*
     * =====================================================
     * ATTEMPTS
     * =====================================================
     */


    for (
        let attempt = 1;

        attempt <= MAX_EXECUTION_ATTEMPTS;

        attempt++
    ) {


        context.attempt =
            attempt;


        console.log(
            `Jessica execution attempt ${attempt}/${MAX_EXECUTION_ATTEMPTS}`
        );


        /*
         * =================================================
         * 1. RUN PLAN
         * =================================================
         */


        try {


            context.runResult =
                await runPlan(

                    context.plan,

                    context.task

                );


        } catch (error) {


            console.error(
                "Jessica TaskRunner exception:",
                error
            );


            return buildFailureResult(

                context,

                {

                    stage:
                        "runner",

                    reason:
                        error?.message ||
                        "Ошибка выполнения плана",

                    failureType:
                        "runner-exception"

                }

            );

        }


        /*
         * =================================================
         * 2. ANALYZE RUN RESULT
         * =================================================
         */


        const runFailure =
            analyzeRunFailure(
                context.runResult
            );


        if (
            runFailure?.failed === true
        ) {


            lastFailure =
                runFailure;


            /*
             * =============================================
             * NEEDS CLARIFICATION
             * =============================================
             */


            if (
                runFailure.needsClarification === true
            ) {

                return buildClarificationResult(

                    context,

                    {

                        stage:
                            runFailure.stage ||
                            "tools",

                        reason:
                            runFailure.reason ||
                            "Для выполнения задачи требуется уточнение"

                    }

                );

            }


            /*
             * =============================================
             * NON-RETRYABLE RUN FAILURE
             * =============================================
             */


            if (
                runFailure.shouldRetry !== true
            ) {

                return buildFailureResult(

                    context,

                    {

                        stage:
                            runFailure.stage ||
                            "tools",

                        reason:
                            runFailure.reason ||
                            "Не удалось выполнить план",

                        failureType:
                            runFailure.failureType ||
                            "run-failure"

                    }

                );

            }


            /*
             * =============================================
             * LAST ATTEMPT
             * =============================================
             */


            if (
                attempt >=
                MAX_EXECUTION_ATTEMPTS
            ) {

                return buildTerminalResult(

                    context,

                    runFailure

                );

            }


            /*
             * =============================================
             * REPLAN AFTER RUN FAILURE
             * =============================================
             */


            const replan =
                await tryReplan(

                    context,

                    buildRunFailureFeedback(
                        runFailure
                    )

                );


            if (
                !replan.success
            ) {

                return replan.result;

            }


            continue;

        }


        /*
         * =================================================
         * 3. COMPOSE ANSWER
         * =================================================
         */


        try {


            context.answerResult =
                await composeAnswer(

                    context.task,

                    context.plan,

                    context.runResult

                );


        } catch (error) {


            console.error(
                "Jessica Answer Composer exception:",
                error
            );


            return buildFailureResult(

                context,

                {

                    stage:
                        "composer",

                    reason:
                        error?.message ||
                        "Ошибка формирования ответа",

                    failureType:
                        "composer-exception"

                }

            );

        }


        if (
            !context.answerResult?.success
        ) {

            return buildFailureResult(

                context,

                {

                    stage:
                        "composer",

                    reason:
                        context.answerResult?.text ||
                        "Ответ не создан",

                    failureType:
                        "composer-failure"

                }

            );

        }


        /*
         * =================================================
         * 4. VALIDATE ANSWER
         * =================================================
         */


        let validation;


        try {


            validation =
                await validateResult(

                    context.task,

                    context.plan,

                    context.runResult,

                    context.answerResult

                );


        } catch (error) {


            /*
             * Validator технически недоступен.
             *
             * Уже сформированный ответ не теряем,
             * но явно отмечаем, что он не был проверен.
             */


            console.error(
                "Jessica Validator exception:",
                error
            );


            return buildCompletedResult(

                context,

                context.answerResult,

                false

            );

        }


        /*
         * =================================================
         * VALID
         * =================================================
         */


        if (
            validation?.valid === true
        ) {

            return buildCompletedResult(

                context,

                context.answerResult,

                true

            );

        }


        /*
         * =================================================
         * NEEDS CLARIFICATION
         * =================================================
         */


        if (
            validation?.needsClarification === true
        ) {

            return buildClarificationResult(

                context,

                {

                    stage:
                        "validator",

                    reason:
                        validation?.reason ||
                        "Для выполнения задачи требуется уточнение"

                }

            );

        }


        /*
         * =================================================
         * VALIDATION FAILURE
         * =================================================
         */


        const validatorFailure =
            buildValidatorFeedback(
                validation
            );


        lastFailure =
            validatorFailure;


        /*
         * =================================================
         * RETRY NOT ALLOWED
         * =================================================
         */


        if (
            !shouldRetryExecution(

                validation,

                attempt

            )
        ) {

            return buildFailureResult(

                context,

                {

                    stage:
                        "validator",

                    reason:
                        validation?.reason ||
                        "Ответ не прошёл проверку",

                    failureType:
                        "validation-failure"

                }

            );

        }


        /*
         * =================================================
         * REPLAN AFTER VALIDATION FAILURE
         * =================================================
         */


        const replan =
            await tryReplan(

                context,

                validatorFailure

            );


        if (
            !replan.success
        ) {

            return replan.result;

        }

    }


    /*
     * =====================================================
     * FALLBACK
     * =====================================================
     *
     * Обычно цикл завершится раньше.
     *
     * Этот блок нужен только как страховка.
     *
     * =====================================================
     */


    return buildTerminalResult(

        context,

        lastFailure || {

            stage:
                "execution",

            failureType:
                "execution-limit",

            reason:
                "Исчерпан лимит выполнения"

        }

    );

}
