/*
 * =========================================================
 * JESSICA EXECUTION CYCLE v3
 * =========================================================
 *
 * Центральный координатор Execution Plan.
 *
 *
 * Flow:
 *
 * Plan
 *   ↓
 * Execution Context
 *   ↓
 * Execution Step Runner
 *   ↓
 * Result
 *
 *
 * Failure:
 *
 * Failure Handler
 *   ↓
 * Retry / Replan
 *
 *
 * НЕ:
 *
 * - выполняет инструменты;
 * - создаёт ответы;
 * - валидирует результат;
 * - сохраняет Experience.
 *
 * =========================================================
 */



import {
    createExecutionContext
} from "./executionContext.js";


import {
    executeExecutionStep
} from "./executionStepRunner.js";


import {
    handleExecutionFailure
} from "./executionFailureHandler.js";


import {
    buildTerminalResult
} from "./executionTerminal.js";


import {
    createExecutionTrace,
    updateTraceFromResult,
    finishExecutionTrace
} from "./executionTrace.js";


import {
    MAX_EXECUTION_ATTEMPTS
} from "./retryPolicy.js";







/*
 * =========================================================
 * BUILD EXPERIENCE META
 * =========================================================
 */


function extractExperience(
    plan,
    planningContext
) {


    return {


        found:

            planningContext
                ?.experience
                ?.found === true,



        source:

            planningContext
                ?.experience
                ?.source ||
            null,



        confidence:

            Number(
                planningContext
                    ?.experience
                    ?.confidence || 0
            ),



        skills:

            Array.isArray(
                plan?.experience?.skills
            )

                ? plan.experience.skills

                : [],



        context:

            planningContext
                ?.experience ||
            null


    };

}









/*
 * =========================================================
 * BUILD FINAL META
 * =========================================================
 */


function buildExecutionMeta(
    context
) {


    return {


        executionId:
            context.executionId,



        attempts:
            context.attempt,



        experienceUsed:

            context
                ?.experience
                ?.skills
                ?.length > 0,



        experienceSkills:

            context
                ?.experience
                ?.skills || [],



        traceId:

            context
                ?.trace
                ?.id ||
            null


    };

}









/*
 * =========================================================
 * SUCCESS RESULT
 * =========================================================
 */


function attachExecutionMeta(
    result,
    context
) {


    return {


        ...result,


        executionMeta:

            buildExecutionMeta(
                context
            )


    };

}









/*
 * =========================================================
 * REGISTER FAILURE
 * =========================================================
 */


function registerFailure(
    context,
    failure
) {


    if (
        !failure
    ) {

        return;

    }



    context.errors.push({

        ...failure,

        timestamp:
            new Date()
                .toISOString()

    });


}









/*
 * =========================================================
 * CREATE EXECUTION CONTEXT
 * =========================================================
 */


function createJessicaContext({

    task,

    plan,

    planningContext

}) {


    return createExecutionContext({

        task,


        plan,


        planningContext,


        experience:

            extractExperience(

                plan,

                planningContext

            )

    });


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



    /*
     * =====================================================
     * CONTEXT
     * =====================================================
     */


    const context =

        createJessicaContext({

            task:
                taskText,


            plan:
                initialPlan,


            planningContext

        });








    /*
     * =====================================================
     * TRACE
     * =====================================================
     */


    context.trace =

        createExecutionTrace(

            taskText

        );






    let lastFailure =
        null;








    /*
     * =====================================================
     * EXECUTION LOOP
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

            "Jessica Execution Cycle:",

            {

                executionId:
                    context.executionId,


                attempt,


                max:
                    MAX_EXECUTION_ATTEMPTS

            }

        );








        /*
         * =================================================
         * STEP RUNNER
         * =================================================
         */


        let stepResult;



        try {


            stepResult =

                await executeExecutionStep(

                    context

                );


        } catch(error) {


            stepResult = {


                success:false,


                failure:{

                    stage:
                        "execution",


                    failureType:
                        "exception",


                    reason:
                        error.message

                }


            };


        }








        /*
         * =================================================
         * TRACE
         * =================================================
         */


        updateTraceFromResult(

            context.trace,

            stepResult

        );









        /*
         * =================================================
         * SUCCESS
         * =================================================
         */


        if (

            stepResult?.success === true

        ) {



            context.state =
                "COMPLETED";



            finishExecutionTrace(
                context.trace
            );



            return attachExecutionMeta(

                stepResult.result,

                context

            );


        }








        /*
         * =================================================
         * FAILURE
         * =================================================
         */


        lastFailure =

            stepResult?.failure || {


                stage:
                    "execution",


                failureType:
                    "unknown",


                reason:
                    "Неизвестная ошибка выполнения"


            };





        registerFailure(

            context,

            lastFailure

        );








        /*
         * =================================================
         * LIMIT
         * =================================================
         */


        if (

            attempt >= MAX_EXECUTION_ATTEMPTS

        ) {



            context.state =
                "FAILED";



            finishExecutionTrace(
                context.trace
            );



            return attachExecutionMeta(

                buildTerminalResult(

                    context,

                    lastFailure

                ),

                context

            );


        }








        /*
         * =================================================
         * FAILURE HANDLER
         * =================================================
         */


        const failureResult =

            await handleExecutionFailure(

                context,

                lastFailure

            );







        if (

            failureResult?.finished === true

        ) {



            context.state =
                failureResult
                    ?.result
                    ?.status ||
                "FAILED";



            finishExecutionTrace(
                context.trace
            );



            return attachExecutionMeta(

                failureResult.result,

                context

            );


        }



    }









    /*
     * =====================================================
     * FALLBACK
     * =====================================================
     */


    context.state =
        "FAILED";


    finishExecutionTrace(
        context.trace
    );



    return attachExecutionMeta(

        buildTerminalResult(

            context,

            lastFailure || {


                stage:
                    "execution",


                failureType:
                    "execution-limit",


                reason:
                    "Лимит выполнения исчерпан"


            }

        ),

        context

    );


}
