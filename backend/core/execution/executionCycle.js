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
 * Центральный цикл выполнения одной задачи.
 *
 *
 * Поток:
 *
 * Plan
 *  ↓
 * TaskRunner
 *  ↓
 * Run Analysis
 *  ↓
 * Answer Composer
 *  ↓
 * Validator
 *  ↓
 * Replanner
 *  ↓
 * Retry
 *
 *
 * Context сохраняется между попытками:
 *
 * Experience
 * PlanningContext
 * Constraints
 * Instructions
 *
 * =========================================================
 */


/*
 * =========================================================
 * COLLECT USED TOOLS
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
        .filter(
            Boolean
        );

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
            answerResult.source ||
            "unknown",


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
 * FAILURE RESULT
 * =========================================================
 */


function buildFailure(
    {
        stage,
        text,
        plan,
        taskRunResult,
        attempt,
        shouldRetry = false,
        failureType = null
    }
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
 * CREATE ALTERNATIVE PLAN
 * =========================================================
 *
 * Единая точка Replan.
 *
 * Сохраняем:
 *
 * - текущий план;
 * - ошибку;
 * - результаты выполнения;
 * - Experience context.
 *
 * =========================================================
 */


async function createAlternativePlan(
    taskText,
    currentPlan,
    failureContext,
    taskRunResult,
    planningContext,
    attempt
) {


    console.log(
        "Jessica Replan requested:",
        JSON.stringify({

            attempt,

            stage:
                failureContext?.stage ||
                "",


            failureType:
                failureContext?.failureType ||
                "",


            reason:
                failureContext?.reason ||
                ""

        })
    );



    try {


        const result =
            await replanTask(

                taskText,

                currentPlan,

                failureContext,

                taskRunResult,

                planningContext

            );



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



        return {

            success:
                true,

            plan:
                result.plan,


            context:
                result.context ||
                planningContext

        };


    } catch(error) {


        console.error(
            "Jessica Replanner error:",
            error
        );


        return {

            success:
                false,

            reason:
                "Ошибка построения альтернативного плана"

        };

    }

}



/*
 * =========================================================
 * EXECUTE PLAN CYCLE
 * =========================================================
 */


export async function executePlanCycle(
    taskText,
    initialPlan,
    planningContext = {}
) {


    let currentPlan =
        initialPlan;


    let currentContext =
        planningContext;


    let lastRunResult =
        null;


    let lastFailure =
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


        } catch(error) {


            console.error(
                "Jessica TaskRunner error:",
                error
            );


            return buildFailure({

                stage:
                    "runner",

                text:
                    "Ошибка выполнения плана",

                plan:
                    currentPlan,

                taskRunResult:
                    lastRunResult,

                attempt

            });

        }



        lastRunResult =
            taskRunResult;



        /*
         * =================================================
         * 2. ANALYZE RUN
         * =================================================
         */


        const runFailure =
            analyzeRunFailure(
                taskRunResult
            );



        if (
            runFailure.failed === true
        ) {


            /*
             * Требуется уточнение
             */


            if (
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
                        runFailure.reason,

                    plan:
                        currentPlan,

                    toolResults:
                        taskRunResult?.results || [],

                    attempt

                };

            }



            /*
             * Можно перестроить маршрут
             */


            if (
                runFailure.shouldRetry === true
            ) {


                if (
                    attempt >= MAX_EXECUTION_ATTEMPTS
                ) {

                    return buildFailure({

                        stage:
                            runFailure.stage,

                        text:
                            runFailure.reason,

                        plan:
                            currentPlan,

                        taskRunResult,

                        attempt,

                        failureType:
                            runFailure.failureType

                    });

                }



                const alternative =
                    await createAlternativePlan(

                        taskText,

                        currentPlan,

                        buildRunFailureFeedback(
                            runFailure
                        ),

                        taskRunResult,

                        currentContext,

                        attempt

                    );



                if (
                    !alternative.success
                ) {

                    return buildFailure({

                        stage:
                            "replanner",

                        text:
                            alternative.reason,

                        plan:
                            currentPlan,

                        taskRunResult,

                        attempt

                    });

                }



                currentPlan =
                    alternative.plan;


                currentContext =
                    alternative.context;



                continue;

            }



            return buildFailure({

                stage:
                    runFailure.stage ||
                    "tools",

                text:
                    runFailure.reason,

                plan:
                    currentPlan,

                taskRunResult,

                attempt,

                failureType:
                    runFailure.failureType

            });

        }



        /*
         * =================================================
         * 3. COMPOSE
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


        } catch(error) {


            return buildFailure({

                stage:
                    "composer",

                text:
                    "Ошибка формирования ответа",

                plan:
                    currentPlan,

                taskRunResult,

                attempt

            });

        }



        if (
            !answerResult?.success
        ) {

            return buildFailure({

                stage:
                    "composer",

                text:
                    answerResult?.text ||
                    "Ответ не создан",

                plan:
                    currentPlan,

                taskRunResult,

                attempt

            });

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


        } catch(error) {


            console.error(
                "Jessica Validator error:",
                error
            );


            return buildCompletedResult(

                currentPlan,

                taskRunResult,

                answerResult,

                attempt

            );

        }



        /*
         * =================================================
         * VALID
         * =================================================
         */


        if (
            validation.valid === true
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
         * NEED CLARIFICATION
         * =================================================
         */


        if (
            validation.needsClarification === true
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
                    validation.reason,

                plan:
                    currentPlan,

                toolResults:
                    taskRunResult.results || [],

                attempt

            };

        }



        /*
         * =================================================
         * RETRY VALIDATOR
         * =================================================
         */


        if (
            !shouldRetryExecution(
                validation,
                attempt
            )
        ) {

            return buildFailure({

                stage:
                    "validator",

                text:
                    validation.reason ||
                    "Ответ не прошёл проверку",

                plan:
                    currentPlan,

                taskRunResult,

                attempt,

                failureType:
                    "validation-failure"

            });

        }



        const alternative =
            await createAlternativePlan(

                taskText,

                currentPlan,

                validation,

                taskRunResult,

                currentContext,

                attempt

            );



        if (
            !alternative.success
        ) {

            return buildFailure({

                stage:
                    "replanner",

                text:
                    alternative.reason,

                plan:
                    currentPlan,

                taskRunResult,

                attempt

            });

        }



        currentPlan =
            alternative.plan;


        currentContext =
            alternative.context;


    }



    /*
     * =====================================================
     * FALLBACK
     * =====================================================
     */


    return buildFailure({

        stage:
            "execution",

        text:
            lastFailure?.reason ||
            "Исчерпан лимит выполнения",

        plan:
            currentPlan,

        taskRunResult:
            lastRunResult,

        attempt:
            MAX_EXECUTION_ATTEMPTS

    });

}
