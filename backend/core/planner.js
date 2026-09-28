/*
 * =========================================================
 * JESSICA PLANNER CORE v4
 * =========================================================
 *
 * Центральный координатор Planner.
 *
 *
 * Flow:
 *
 * Task
 *   ↓
 * Planner Context
 *   ↓
 * Planner Attempt
 *   ↓
 * Retry Policy
 *   ↓
 * Execution Plan
 *
 *
 * Этот файл отвечает только за:
 *
 * - orchestration;
 * - retry loop;
 * - итоговый результат.
 *
 *
 * Детали вынесены в:
 *
 * planner/runtime/plannerContext.js
 * planner/runtime/plannerAttempt.js
 * planner/runtime/plannerPlanResult.js
 * planner/plannerRetry.js
 *
 * =========================================================
 */


import {

    buildPlannerContext,

    buildPlannerAttemptContext

} from "./planner/runtime/plannerContext.js";


import {
    runPlannerAttempt
} from "./planner/runtime/plannerAttempt.js";


import {
    enrichPlan
} from "./planner/runtime/plannerPlanResult.js";


import {

    MAX_PLANNER_ATTEMPTS,

    sleep,

    isRetryablePlannerError,

    isTechnicalPlannerError,

    isSemanticPlannerError,

    getPlannerRetryDelay

} from "./planner/plannerRetry.js";


/*
 * =========================================================
 * CREATE PLAN
 * =========================================================
 */


export async function createPlan(

    task,

    context = {}

) {


    const cleanTask =
        String(
            task || ""
        )
        .trim();


    /*
     * =====================================================
     * INPUT
     * =====================================================
     */


    if (
        !cleanTask
    ) {


        return {

            success:
                false,

            text:
                "Задача Planner пустая"

        };

    }


    if (
        !process.env.GROQ_API_KEY
    ) {


        return {

            success:
                false,

            text:
                "GROQ_API_KEY отсутствует"

        };

    }


    /*
     * =====================================================
     * BASE CONTEXT
     * =====================================================
     */


    const baseContext =
        buildPlannerContext(
            context
        );


    /*
     * Только semantic feedback
     * передаётся обратно AI.
     */

    let plannerFeedback =
        "";


    let lastError =
        "";


    const trace =
        [];


    /*
     * =====================================================
     * RETRY LOOP
     * =====================================================
     */


    for (

        let attempt = 1;

        attempt <= MAX_PLANNER_ATTEMPTS;

        attempt++

    ) {


        const planningContext =
            buildPlannerAttemptContext(

                baseContext,

                attempt

            );


        try {


            /*
             * =================================================
             * ONE ATTEMPT
             * =================================================
             */


            const attemptResult =
                await runPlannerAttempt({

                    task:
                        cleanTask,

                    feedback:
                        plannerFeedback,

                    context:
                        planningContext,

                    attempt

                });


            /*
             * =================================================
             * SEMANTIC VALIDATION FAILURE
             * =================================================
             */


            if (
                !attemptResult.success
            ) {


                lastError =

                    attemptResult.feedback ||

                    "planner_validation_failed";


                plannerFeedback =
                    lastError;


                trace.push({

                    stage:
                        attemptResult.stage ||
                        "validation",

                    attempt,

                    success:
                        false,

                    type:
                        "semantic",

                    reason:
                        lastError

                });


                continue;

            }


            /*
             * =================================================
             * SUCCESS
             * =================================================
             */


            const finalPlan =
                enrichPlan(

                    attemptResult.plan,

                    planningContext,

                    attempt

                );


            trace.push({

                stage:
                    "completed",

                attempt,

                success:
                    true

            });


            console.log(

                "Jessica Planner completed:",

                {

                    intent:
                        finalPlan.intent,

                    steps:
                        finalPlan.steps.length,

                    attempt

                }

            );


            return {

                success:
                    true,

                plan:
                    finalPlan,

                context:
                    planningContext,

                trace

            };


        } catch(error) {


            /*
             * =================================================
             * ERROR CLASSIFICATION
             * =================================================
             */


            const errorMessage =

                error?.message ||

                "planner_error";


            lastError =
                errorMessage;


            const technical =
                isTechnicalPlannerError(
                    error
                );


            const semantic =
                isSemanticPlannerError(
                    error
                );


            const retryable =
                isRetryablePlannerError(
                    error
                );


            /*
             * Semantic error:
             *
             * модель получила запрос,
             * но сформировала плохой результат.
             *
             * Поэтому feedback нужен.
             */


            if (
                semantic
            ) {


                plannerFeedback =
                    errorMessage;

            }


            /*
             * Technical error:
             *
             * 429 / 5xx / network.
             *
             * Это НЕ ошибка маршрута,
             * поэтому AI не должен её видеть.
             */


            if (
                technical
            ) {


                plannerFeedback =
                    "";

            }


            /*
             * =================================================
             * RETRY DELAY
             * =================================================
             */


            const canRetry =

                retryable &&

                attempt <
                MAX_PLANNER_ATTEMPTS;


            const delay =

                canRetry

                    ? getPlannerRetryDelay(
                        attempt,
                        error
                    )

                    : 0;


            /*
             * =================================================
             * TRACE
             * =================================================
             */


            trace.push({

                stage:
                    "error",

                attempt,

                success:
                    false,

                error:
                    errorMessage,

                type:

                    technical

                        ? "technical"

                        : semantic

                            ? "semantic"

                            : "non-retryable",

                retryable,

                delay

            });


            console.error(

                "Jessica Planner error:",

                {

                    attempt,

                    type:

                        technical

                            ? "technical"

                            : semantic

                                ? "semantic"

                                : "non-retryable",

                    retryable,

                    retryDelay:
                        delay,

                    error:
                        errorMessage

                }

            );


            /*
             * =================================================
             * RETRY
             * =================================================
             */


            if (
                canRetry
            ) {


                if (
                    delay > 0
                ) {


                    await sleep(
                        delay
                    );

                }


                continue;

            }


            break;

        }

    }


    /*
     * =====================================================
     * FAILED
     * =====================================================
     */


    return {

        success:
            false,

        text:

            "Planner не смог создать корректный план: "
            +
            lastError,

        trace

    };

}


/*
 * =========================================================
 * LEGACY EXPORT
 * =========================================================
 */


export async function planTask(

    task,

    context = {}

) {


    const result =
        await createPlan(

            task,

            context

        );


    if (
        result.success
    ) {


        return result.plan;

    }


    throw new Error(
        result.text
    );

}
