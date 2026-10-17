/*
 * =========================================================
 * JESSICA PLANNER CORE v7
 * =========================================================
 *
 * Task
 *      ↓
 * Planner Attempt
 *      ↓
 * Machine Validation
 *      ↓
 *
 * success
 *      → Execution Plan
 *
 * semantic rejection
 *      → Structured Repair Feedback
 *      → Next Planner Attempt
 *
 *
 * Technical AI retry остаётся
 * отдельным механизмом.
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
    buildPlannerAttemptFeedback
} from "./planner/runtime/plannerAttemptFeedback.js";


import {

    createPlannerRepairMemory,

    rememberPlannerRejection,

    buildPlannerRepairMemoryText

} from "./planner/runtime/plannerRepairMemory.js";


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


    if(
        !cleanTask
    ){

        return {

            success:
                false,

            text:
                "Задача Planner пустая",

            trace:
                []

        };

    }


    if(
        !process.env.GROQ_API_KEY
    ){

        return {

            success:
                false,

            text:
                "GROQ_API_KEY отсутствует",

            trace:
                []

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
     * Semantic feedback
     * следующей AI попытке.
     */


    let plannerFeedback =
        "";


    const repairMemory =
        createPlannerRepairMemory();


    let lastError =
        "";


    const trace =
        [];


    /*
     * =====================================================
     * ATTEMPT LOOP
     * =====================================================
     */


    for(

        let attempt = 1;

        attempt <= MAX_PLANNER_ATTEMPTS;

        attempt++

    ){

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


            if(
                !attemptResult.success
            ){

                lastError =

                    attemptResult.feedback
                    ||
                    "planner_validation_failed";


                const traceEntry = {

                    stage:

                        attemptResult.stage
                        ||
                        "validation",

                    attempt,

                    success:
                        false,

                    type:
                        "semantic",

                    reason:
                        lastError

                };


                trace.push(
                    traceEntry
                );


                console.warn(

                    "Jessica Planner semantic rejection:",

                    {

                        attempt,

                        stage:
                            traceEntry.stage,

                        reason:
                            lastError

                    }

                );


                /*
                 * =================================================
                 * STRUCTURED REPAIR
                 * =================================================
                 *
                 * Только если впереди
                 * действительно есть новая попытка.
                 * =================================================
                 */


                if(
                    attempt
                    <
                    MAX_PLANNER_ATTEMPTS
                ){

                    rememberPlannerRejection(

                        repairMemory,

                        lastError

                    );


                    const currentRepair =

                        buildPlannerAttemptFeedback({

                            reason:
                                lastError,

                            rejectedPlan:
                                attemptResult.rejectedPlan,

                            attempt

                        });


                    const persistentRepair =

                        buildPlannerRepairMemoryText(
                            repairMemory
                        );


                    plannerFeedback =

                        [

                            persistentRepair,

                            currentRepair

                        ]
                            .filter(Boolean)
                            .join(
                                "\n\n"
                            );


                    console.warn(

                        "Jessica Planner semantic repair prepared:",

                        {

                            rejectedAttempt:
                                attempt,

                            nextAttempt:
                                attempt + 1,

                            feedback:
                                plannerFeedback

                        }

                    );

                }


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

                    evidenceMode:

                        finalPlan
                            ?.evidence
                            ?.mode

                        ||

                        "none",

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

        }
        catch(error){


            /*
             * =================================================
             * ERROR CLASSIFICATION
             * =================================================
             */


            const errorMessage =

                error?.message
                ||
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
             * =================================================
             * SEMANTIC EXCEPTION
             * =================================================
             */


            if(
                semantic
            ){

                rememberPlannerRejection(

                    repairMemory,

                    errorMessage

                );


                plannerFeedback =

                    [

                        buildPlannerRepairMemoryText(
                            repairMemory
                        ),

                        buildPlannerAttemptFeedback({

                            reason:
                                errorMessage,

                            rejectedPlan:
                                null,

                            attempt

                        })

                    ]
                        .filter(Boolean)
                        .join(
                            "\n\n"
                        );

            }


            /*
             * =================================================
             * TECHNICAL ERROR
             * =================================================
             *
             * Technical provider failure
             * is not semantic feedback.
             * =================================================
             */


            if(
                technical
            ){

                /*
                 * Provider failure не создаёт новую
                 * semantic instruction, но уже
                 * накопленные machine constraints
                 * терять нельзя.
                 */

                plannerFeedback =

                    buildPlannerRepairMemoryText(
                        repairMemory
                    );

            }


            /*
             * =================================================
             * RETRY DECISION
             * =================================================
             */


            const canRetry =

                retryable

                &&

                attempt
                <
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


            if(
                canRetry
            ){

                if(
                    delay > 0
                ){

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
     * EXHAUSTED
     * =====================================================
     */


    const failureText =

        "Planner не смог создать корректный план: "

        +

        (
            lastError
            ||
            "unknown planner failure"
        );


    console.error(

        "Jessica Planner exhausted attempts:",

        {

            attempts:
                trace.length,

            lastError:

                lastError
                ||
                null,

            trace

        }

    );


    return {

        success:
            false,

        text:
            failureText,

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


    if(
        result.success
    ){

        return result.plan;

    }


    throw new Error(
        result.text
    );

}
