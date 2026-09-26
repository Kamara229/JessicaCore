/*
 * =========================================================
 * JESSICA EXECUTION CYCLE v5
 * =========================================================
 *
 * Главный координатор Execution Run.
 *
 *
 * Flow:
 *
 * Plan
 *   ↓
 * Execution Context
 *   ↓
 * Execution Trace
 *   ↓
 * Step Runner
 *   ↓
 * Failure Handler
 *   ↓
 * Result
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
    finishExecutionTrace,
    addTraceEvent
} from "../trace/executionTrace.js";


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


    const experience =
        planningContext?.experience;



    return {


        found:

            experience?.found === true,



        source:

            experience?.source ||
            null,



        confidence:

            Number(
                experience?.confidence || 0
            ),



        skills:

            Array.isArray(
                plan?.experience?.skills
            )

                ? plan.experience.skills

                :

                [],



        context:

            experience || null


    };

}









/*
 * =========================================================
 * CONTEXT
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
 * META
 * =========================================================
 */


function attachExecutionMeta(

    result,

    context

) {


    return {


        ...result,


        executionMeta:

        {


            executionId:

                context.executionId,



            traceId:

                context.trace?.id || null,



            attempts:

                context.attempt,



            replans:

                context.replanCount,



            experience:

            {


                used:

                    context.experience.found === true,



                skills:

                    context.experience.skills || []

            }


        }


    };

}









/*
 * =========================================================
 * ERROR REGISTER
 * =========================================================
 */


function registerFailure(

    context,

    failure

) {


    context.errors.push({

        ...failure,


        timestamp:

            new Date()
                .toISOString()

    });


}









/*
 * =========================================================
 * MAIN EXECUTION
 * =========================================================
 */


export async function executePlanCycle(

    task,

    initialPlan,

    planningContext = {}

) {


    const context =

        createJessicaContext({

            task,

            plan:
                initialPlan,

            planningContext

        });








    /*
     * TRACE CREATE
     */


    context.trace =

        createExecutionTrace(

            task

        );





    addTraceEvent(

        context.trace,

        "EXECUTION_STARTED",

        {

            executionId:
                context.executionId

        }

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



        addTraceEvent(

            context.trace,

            "ATTEMPT_STARTED",

            {

                attempt

            }

        );








        let result;





        try {


            result =

                await executeExecutionStep(

                    context

                );



        } catch(error) {



            result = {


                success:false,


                failure:{

                    stage:
                        "execution",


                    failureType:
                        "exception",


                    reason:

                        error?.message ||

                        "Execution error"

                }


            };


        }








        /*
         * TRACE
         */


        updateTraceFromResult(

            context.trace,

            result

        );









        /*
         * SUCCESS
         */


        if (

            result?.success === true

        ) {



            context.state =
                "COMPLETED";



            addTraceEvent(

                context.trace,

                "EXECUTION_COMPLETED"

            );



            finishExecutionTrace(

                context.trace

            );



            return attachExecutionMeta(

                result.result,

                context

            );

        }









        /*
         * FAILURE
         */


        lastFailure =

            result?.failure ||

            {

                stage:
                    "execution",


                failureType:
                    "unknown",


                reason:
                    "Unknown failure"

            };




        registerFailure(

            context,

            lastFailure

        );









        /*
         * TERMINAL LIMIT
         */


        if (

            attempt >= MAX_EXECUTION_ATTEMPTS

        ) {


            context.state =
                "FAILED";



            const terminal =

                buildTerminalResult(

                    context,

                    lastFailure

                );



            finishExecutionTrace(

                context.trace

            );



            return attachExecutionMeta(

                terminal,

                context

            );


        }









        /*
         * FAILURE HANDLER
         */


        const decision =

            await handleExecutionFailure(

                context,

                lastFailure

            );









        if (

            decision?.action === "REPLAN"

        ) {


            context.replanCount++;


            addTraceEvent(

                context.trace,

                "REPLAN_REQUESTED",

                decision

            );


        }








        if (

            decision?.finished === true

        ) {


            context.state =
                "FAILED";



            finishExecutionTrace(

                context.trace

            );



            return attachExecutionMeta(

                decision.result,

                context

            );

        }






    }









    /*
     * FALLBACK
     */


    finishExecutionTrace(

        context.trace

    );



    return attachExecutionMeta(

        buildTerminalResult(

            context,

            lastFailure || {

                stage:
                    "execution",

                reason:
                    "Execution limit reached"

            }

        ),

        context

    );


}
