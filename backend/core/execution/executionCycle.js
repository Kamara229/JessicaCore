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
 * Центральный исполнитель Jessica.
 *
 *
 * Flow:
 *
 * PlanningContext
 *        |
 *        v
 *      Plan
 *        |
 *        v
 *    TaskRunner
 *        |
 *        v
 *     Analyze
 *        |
 *        v
 * Answer Composer
 *        |
 *        v
 *    Validator
 *        |
 *        v
 *    Replanner
 *        |
 *        v
 *      Retry
 *
 *
 * Execution Cycle НЕ:
 *
 * - создаёт планы;
 * - ищет Experience;
 * - выполняет AI planning;
 * - хранит обучение.
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
 * COMPLETED RESULT
 * =========================================================
 */


function buildCompletedResult(
    context,
    answerResult,
    validated
) {

    return {

        success:
            true,


        status:
            "COMPLETED",


        validated,


        validationStatus:
            validated
                ? "passed"
                : "skipped",


        result:
            answerResult.text,


        answerSource:
            answerResult.source ||
            "unknown",


        usedTools:
            collectUsedTools(
                context.runResult
            ),


        plan:
            context.plan,


        planningContext:
            context.planningContext,


        toolResults:
            context.runResult?.results || [],


        attempt:
            context.attempt

    };

}



/*
 * =========================================================
 * FAILURE RESULT
 * =========================================================
 */


function buildFailure(
    context,
    stage,
    reason,
    failureType = null
) {

    return {

        success:
            false,


        status:
            "FAILED",


        stage,


        failureType,


        shouldRetry:
            false,


        result:
            reason,


        plan:
            context.plan,


        planningContext:
            context.planningContext,


        toolResults:
            context.runResult?.results || [],


        attempt:
            context.attempt

    };

}



/*
 * =========================================================
 * NEEDS CLARIFICATION
 * =========================================================
 */


function buildClarificationResult(
    context,
    stage,
    reason
) {

    return {

        success:
            false,


        status:
            "NEEDS_CLARIFICATION",


        needsClarification:
            true,


        stage,


        result:
            reason,


        plan:
            context.plan,


        planningContext:
            context.planningContext,


        toolResults:
            context.runResult?.results || [],


        attempt:
            context.attempt

    };

}



/*
 * =========================================================
 * CREATE REPLAN
 * =========================================================
 */


async function createAlternativePlan(
    context,
    feedback
) {


    console.log(
        "Jessica Replan requested:",
        JSON.stringify({

            attempt:
                context.attempt,

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

                context.task,

                context.plan,

                feedback,

                context.runResult,

                context.planningContext

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


            planningContext:
                result.context ||
                context.planningContext

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


        } catch(error) {


            console.error(
                "Jessica TaskRunner error:",
                error
            );


            return buildFailure(

                context,

                "runner",

                "Ошибка выполнения плана"

            );

        }



        /*
         * =================================================
         * 2. ANALYZE RUN
         * =================================================
         */


        const runFailure =
            analyzeRunFailure(
                context.runResult
            );



        if (
            runFailure.failed === true
        ) {


            lastFailure =
                runFailure;



            if (
                runFailure.needsClarification === true
            ) {

                return buildClarificationResult(

                    context,

                    runFailure.stage ||
                    "tools",

                    runFailure.reason

                );

            }



            if (
                runFailure.shouldRetry !== true
            ) {

                return buildFailure(

                    context,

                    runFailure.stage ||
                    "tools",

                    runFailure.reason,

                    runFailure.failureType

                );

            }



            if (
                attempt >= MAX_EXECUTION_ATTEMPTS
            ) {

                return buildFailure(

                    context,

                    runFailure.stage,

                    runFailure.reason,

                    runFailure.failureType

                );

            }



            const alternative =
                await createAlternativePlan(

                    context,

                    buildRunFailureFeedback(
                        runFailure
                    )

                );



            if (
                !alternative.success
            ) {

                return buildFailure(

                    context,

                    "replanner",

                    alternative.reason

                );

            }



            context.plan =
                alternative.plan;


            context.planningContext =
                alternative.planningContext;



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


        } catch(error) {


            return buildFailure(

                context,

                "composer",

                "Ошибка формирования ответа"

            );

        }



        if (
            !context.answerResult?.success
        ) {

            return buildFailure(

                context,

                "composer",

                context.answerResult?.text ||
                "Ответ не создан"

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

                    context.task,

                    context.plan,

                    context.runResult,

                    context.answerResult

                );


        } catch(error) {


            console.error(
                "Jessica Validator error:",
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
            validation.valid === true
        ) {

            return buildCompletedResult(

                context,

                context.answerResult,

                true

            );

        }



        /*
         * =================================================
         * CLARIFICATION
         * =================================================
         */


        if (
            validation.needsClarification === true
        ) {

            return buildClarificationResult(

                context,

                "validator",

                validation.reason

            );

        }



        /*
         * =================================================
         * RETRY POLICY
         * =================================================
         */


        if (
            !shouldRetryExecution(

                validation,

                attempt

            )
        ) {

            return buildFailure(

                context,

                "validator",

                validation.reason,

                "validation-failure"

            );

        }



        /*
         * =================================================
         * REPLAN AFTER VALIDATOR
         * =================================================
         */


        const alternative =
            await createAlternativePlan(

                context,

                validation

            );



        if (
            !alternative.success
        ) {

            return buildFailure(

                context,

                "replanner",

                alternative.reason

            );

        }



        context.plan =
            alternative.plan;


        context.planningContext =
            alternative.planningContext;



    }



    /*
     * =====================================================
     * FALLBACK
     * =====================================================
     */


    return buildFailure(

        context,

        "execution",

        lastFailure?.reason ||
        "Исчерпан лимит выполнения"

    );

}
