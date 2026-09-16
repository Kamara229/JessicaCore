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
 * Центральный цикл выполнения Jessica.
 *
 *
 * Flow:
 *
 * Plan
 *  ↓
 * Run
 *  ↓
 * Analyze
 *  ↓
 * Compose
 *  ↓
 * Validate
 *  ↓
 * Replan
 *  ↓
 * Retry
 *
 *
 * Context сохраняется:
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
        .filter(
            Boolean
        );

}



/*
 * =========================================================
 * COMPLETED
 * =========================================================
 */


function buildCompletedResult(
    plan,
    taskRunResult,
    answerResult,
    attempt,
    context,
    validationSkipped = false
) {

    return {

        success:
            true,


        status:
            "COMPLETED",


        validated:
            !validationSkipped,


        validationSkipped,


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


        planningContext:
            context,


        toolResults:
            taskRunResult?.results || [],


        attempt

    };

}



/*
 * =========================================================
 * FAILURE
 * =========================================================
 */


function buildFailure(
    {
        stage,
        text,
        plan,
        taskRunResult,
        attempt,
        context,
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


        planningContext:
            context,


        toolResults:
            taskRunResult?.results || [],


        attempt

    };

}



/*
 * =========================================================
 * VALIDATOR FEEDBACK
 * =========================================================
 */


function buildValidatorFeedback(
    validation
) {

    return {

        stage:
            "validator",


        failureType:
            "validation-failure",


        reason:
            validation?.reason ||
            "Ответ не прошёл проверку",


        shouldRetry:
            true

    };

}



/*
 * =========================================================
 * REPLAN
 * =========================================================
 */


async function createAlternativePlan(
    taskText,
    currentPlan,
    feedback,
    taskRunResult,
    context,
    attempt
) {


    console.log(
        "Jessica Replan:",
        JSON.stringify({

            attempt,

            stage:
                feedback?.stage,

            failureType:
                feedback?.failureType,

            reason:
                feedback?.reason

        })
    );



    try {


        const result =
            await replanTask(

                taskText,

                currentPlan,

                feedback,

                taskRunResult,

                context

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
                    "Replanner не создал план"

            };

        }



        return {

            success:
                true,


            plan:
                result.plan,


            context:
                result.context ||
                context

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
                error?.message ||
                "Ошибка Replanner"

        };

    }

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
         * RUN
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


            return buildFailure({

                stage:
                    "runner",

                text:
                    "Ошибка выполнения плана",

                plan:
                    currentPlan,

                taskRunResult:
                    lastRunResult,

                attempt,

                context

            });

        }



        lastRunResult =
            taskRunResult;



        /*
         * =================================================
         * RUN ANALYSIS
         * =================================================
         */


        const runFailure =
            analyzeRunFailure(
                taskRunResult
            );



        if (
            runFailure.failed === true
        ) {


            lastFailure =
                runFailure;



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


                    result:
                        runFailure.reason,


                    plan:
                        currentPlan,


                    planningContext:
                        currentContext,


                    toolResults:
                        taskRunResult.results || [],


                    attempt

                };

            }



            if (
                runFailure.shouldRetry !== true
            ) {

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

                    context:
                        currentContext,

                    failureType:
                        runFailure.failureType

                });

            }



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

                    context:
                        currentContext,

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

                    attempt,

                    context:
                        currentContext

                });

            }



            currentPlan =
                alternative.plan;


            currentContext =
                alternative.context;



            continue;

        }



        /*
         * =================================================
         * COMPOSE
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

                attempt,

                context:
                    currentContext

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

                attempt,

                context:
                    currentContext

            });

        }



        /*
         * =================================================
         * VALIDATE
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
                "Validator error:",
                error
            );


            return buildCompletedResult(

                currentPlan,

                taskRunResult,

                answerResult,

                attempt,

                currentContext,

                true

            );

        }



        if (
            validation.valid === true
        ) {

            return buildCompletedResult(

                currentPlan,

                taskRunResult,

                answerResult,

                attempt,

                currentContext

            );

        }



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


                planningContext:
                    currentContext,


                toolResults:
                    taskRunResult.results || [],


                attempt

            };

        }



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
                    validation.reason,

                plan:
                    currentPlan,

                taskRunResult,

                attempt,

                context:
                    currentContext,

                failureType:
                    "validation-failure"

            });

        }



        const alternative =
            await createAlternativePlan(

                taskText,

                currentPlan,

                buildValidatorFeedback(
                    validation
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

                attempt,

                context:
                    currentContext

            });

        }



        currentPlan =
            alternative.plan;


        currentContext =
            alternative.context;



    }



    return buildFailure({

        stage:
            "execution",

        text:
            lastFailure?.reason ||
            "Исчерпан лимит попыток",

        plan:
            currentPlan,

        taskRunResult:
            lastRunResult,

        attempt:
            MAX_EXECUTION_ATTEMPTS,

        context:
            currentContext

    });

}
