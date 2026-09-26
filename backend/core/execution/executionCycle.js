/*
 * =========================================================
 * JESSICA EXECUTION CYCLE v4
 * =========================================================
 *
 * Главный координатор Execution.
 *
 *
 * Flow:
 *
 * Plan
 *   ↓
 * Execution Context
 *   ↓
 * Trace
 *   ↓
 * Step Runner
 *   ↓
 * Result
 *
 *
 * Failure:
 *
 * Failure Handler
 *   ↓
 * Retry / Replan / Terminal
 *
 *
 * НЕ:
 *
 * - выполняет инструменты;
 * - создаёт ответы;
 * - валидирует;
 * - обучает Jessica.
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
 * EXPERIENCE
 * =========================================================
 */


function extractExperience(
    plan,
    planningContext
) {


    return {


        used:

            plan?.experienceUsed === true,



        skills:

            Array.isArray(
                plan?.experience?.skills
            )

                ? plan.experience.skills

                : [],



        context:

            planningContext?.experience ||
            null



    };

}









/*
 * =========================================================
 * META
 * =========================================================
 */


function buildExecutionMeta(
    context
) {


    return {


        executionId:

            context.executionId ||
            null,



        traceId:

            context.trace?.id ||
            null,



        attempts:

            context.attempt || 0,



        experienceUsed:

            context.experience?.used === true,



        experienceSkills:

            context.experience?.skills || []

    };


}









function attachMeta(
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
 * FAILURE LOG
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



    if (
        !Array.isArray(
            context.errors
        )
    ) {

        context.errors = [];

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
 * CREATE CONTEXT
 * =========================================================
 */


function createJessicaExecutionContext({

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
 * EXECUTE CYCLE
 * =========================================================
 */


export async function executePlanCycle(

    task,

    initialPlan,

    planningContext = {}

) {


    const context =

        createJessicaExecutionContext({

            task,

            plan:
                initialPlan,

            planningContext


        });








    context.trace =

        createExecutionTrace(

            task

        );








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

            "Jessica Execution:",

            {

                executionId:
                    context.executionId,


                attempt,


                max:
                    MAX_EXECUTION_ATTEMPTS

            }

        );








        let executionResult;





        /*
         * =================================================
         * STEP RUNNER
         * =================================================
         */


        try {


            executionResult =

                await executeExecutionStep(

                    context

                );


        } catch(error) {


            executionResult = {


                success:false,


                failure:{

                    stage:
                        "execution",


                    failureType:
                        "exception",


                    reason:

                        error?.message ||

                        "Execution exception"

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

            executionResult

        );









        /*
         * =================================================
         * SUCCESS
         * =================================================
         */


        if (

            executionResult?.success === true

        ) {


            context.state =
                "COMPLETED";


            finishExecutionTrace(
                context.trace
            );



            return attachMeta(

                executionResult.result,

                context

            );


        }









        /*
         * =================================================
         * FAILURE
         * =================================================
         */


        lastFailure =

            executionResult?.failure ||

            {

                stage:
                    "execution",


                failureType:
                    "unknown",


                reason:
                    "Unknown execution failure"

            };





        registerFailure(

            context,

            lastFailure

        );









        /*
         * =================================================
         * FAILURE HANDLER
         * =================================================
         */


        const decision =

            await handleExecutionFailure(

                context,

                lastFailure

            );









        /*
         * =================================================
         * TERMINAL
         * =================================================
         */


        if (

            decision?.action ===
            "TERMINAL"

        ) {


            context.state =
                "FAILED";


            finishExecutionTrace(
                context.trace
            );



            return attachMeta(

                decision.result,

                context

            );


        }









        /*
         * =================================================
         * OLD LIMIT PROTECTION
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



            return attachMeta(

                buildTerminalResult(

                    context,

                    {

                        status:
                            "FAILED",


                        stage:
                            lastFailure.stage,


                        reason:
                            lastFailure.reason,


                        failureType:
                            lastFailure.failureType

                    }

                ),

                context

            );


        }









        /*
         * CONTINUE:
         *
         * Failure Handler мог:
         *
         * - обновить Plan;
         * - подготовить Replan;
         * - изменить Context.
         *
         */


    }









    /*
     * =====================================================
     * FALLBACK
     * =====================================================
     */


    finishExecutionTrace(
        context.trace
    );



    return attachMeta(

        buildTerminalResult(

            context,

            {

                status:
                    "FAILED",


                stage:
                    "execution",


                reason:
                    "Execution limit reached",


                failureType:
                    "execution-limit"


            }

        ),

        context

    );


}
