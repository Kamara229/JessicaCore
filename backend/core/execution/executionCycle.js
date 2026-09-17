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
 * Центральный координатор выполнения.
 *
 *
 * Flow:
 *
 * Plan
 *   ↓
 * TaskRunner
 *   ↓
 * Run Analysis
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
 * Вся специализированная логика вынесена в:
 *
 * - executionResult.js
 * - replanCoordinator.js
 * - retryPolicy.js
 * - runFailurePolicy.js
 * - terminalOutcomePolicy.js
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
 * TRY REPLAN
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
 * RUN PLAN
 * =========================================================
 */


async function executeCurrentPlan(
    context
) {

    try {

        context.runResult =
            await runPlan(

                context.plan,

                context.task

            );


        return {

            success:
                true

        };

    } catch (error) {

        console.error(
            "Jessica TaskRunner exception:",
            error
        );


        return {

            success:
                false,

            result:
                buildFailureResult(

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

                )

        };

    }

}


/*
 * =========================================================
 * COMPOSE ANSWER
 * =========================================================
 */


async function composeCurrentAnswer(
    context
) {

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


        return {

            success:
                false,

            result:
                buildFailureResult(

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

                )

        };

    }


    if (
        !context.answerResult?.success
    ) {

        return {

            success:
                false,

            result:
                buildFailureResult(

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
 * VALIDATE ANSWER
 * =========================================================
 */


async function validateCurrentAnswer(
    context
) {

    try {

        const validation =
            await validateResult(

                context.task,

                context.plan,

                context.runResult,

                context.answerResult

            );


        return {

            success:
                true,

            validation

        };

    } catch (error) {

        console.error(
            "Jessica Validator exception:",
            error
        );


        /*
         * Ответ уже сформирован.
         *
         * При технической ошибке Validator
         * не теряем результат,
         * но помечаем его как непроверенный.
         */

        return {

            success:
                false,

            result:
                buildCompletedResult(

                    context,

                    context.answerResult,

                    false

                )

        };

    }

}


/*
 * =========================================================
 * HANDLE VALID RESULT
 * =========================================================
 */


function buildValidResult(
    context,
    validation
) {

    /*
     * =====================================================
     * VALID NEGATIVE OUTCOME
     * =====================================================
     *
     * Ответ корректен,
     * но искомый подтверждённый результат
     * получить не удалось.
     *
     * Это COMPLETED, а не FAILED.
     *
     * =====================================================
     */


    if (
        validation?.outcomeType ===
        "no_verified_result"
    ) {

        console.log(
            "Jessica semantic outcome: no_verified_result"
        );


        return buildNoVerifiedResult(

            context,

            {

                message:
                    context.answerResult?.text ||
                    "Не удалось подтвердить достоверный результат по доступным источникам.",

                reason:
                    validation?.reason ||
                    "Подтверждённый результат не найден",

                stage:
                    "validator",

                failureType:
                    "no-verified-result"

            }

        );

    }


    /*
     * =====================================================
     * NORMAL RESULT
     * =====================================================
     */


    return buildCompletedResult(

        context,

        context.answerResult,

        true

    );

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
     * EXECUTION ATTEMPTS
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


        const execution =
            await executeCurrentPlan(
                context
            );


        if (
            !execution.success
        ) {

            return execution.result;

        }


        /*
         * =================================================
         * 2. ANALYZE RUN FAILURE
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
             * ---------------------------------------------
             * CLARIFICATION
             * ---------------------------------------------
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
             * ---------------------------------------------
             * NON-RETRYABLE FAILURE
             * ---------------------------------------------
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
             * ---------------------------------------------
             * ATTEMPTS EXHAUSTED
             * ---------------------------------------------
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
             * ---------------------------------------------
             * REPLAN
             * ---------------------------------------------
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


        const composition =
            await composeCurrentAnswer(
                context
            );


        if (
            !composition.success
        ) {

            return composition.result;

        }


        /*
         * =================================================
         * 4. VALIDATE ANSWER
         * =================================================
         */


        const validationResult =
            await validateCurrentAnswer(
                context
            );


        if (
            !validationResult.success
        ) {

            return validationResult.result;

        }


        const validation =
            validationResult.validation;


        /*
         * =================================================
         * VALID
         * =================================================
         */


        if (
            validation?.valid === true
        ) {

            return buildValidResult(

                context,

                validation

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
