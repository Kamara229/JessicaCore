/*
 * =========================================================
 * JESSICA EXECUTION CYCLE v6
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
 * Trace
 *   ↓
 * Step Runner
 *   ↓
 * Failure Handler
 *   ↓
 *
 * RETRY
 * REPLAN
 * CLARIFICATION
 * FINISH
 *
 *
 * НЕ:
 *
 * - выполняет Tools;
 * - создаёт Answer;
 * - валидирует;
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
    handleExecutionFailure,
    FAILURE_ACTION
} from "./executionFailureHandler.js";


import {
    createAlternativePlan,
    applyAlternativePlan
} from "./replanCoordinator.js";


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

            experience?.source || null,



        confidence:

            Number(
                experience?.confidence || 0
            ),



        skills:

            Array.isArray(
                plan?.experience?.skills
            )

                ? plan.experience.skills

                : [],



        context:

            experience || null


    };

}









/*
 * =========================================================
 * CREATE CONTEXT
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

                context.executionId || null,



            traceId:

                context.trace?.id || null,



            attempts:

                context.attempt || 0,



            replans:

                context.replanCount || 0,



            experience:

            {

                used:

                    context.experience?.found === true,



                skills:

                    context.experience?.skills || []

            }


        }


    };

}









/*
 * =========================================================
 * FAILURE REGISTER
 * =========================================================
 */


function registerFailure(

    context,

    failure

) {


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
 * HANDLE REPLAN
 * =========================================================
 */


async function processReplan(

    context,

    decision

) {


    addTraceEvent(

        context.trace,

        "REPLAN_STARTED",

        {

            reason:
                decision.reason

        }

    );





    const alternative =

        await createAlternativePlan(

            context,

            decision.failure

        );






    if (
        !alternative.success
    ) {


        addTraceEvent(

            context.trace,

            "REPLAN_FAILED",

            alternative

        );


        return false;

    }






    const applied =

        applyAlternativePlan(

            context,

            alternative

        );





    if (
        !applied
    ) {


        return false;

    }





    context.replanCount =

        Number(
            context.replanCount || 0
        )
        +
        1;





    addTraceEvent(

        context.trace,

        "REPLAN_COMPLETED",

        {

            intent:
                context.plan?.intent

        }

    );



    return true;


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







        let executionResult;



        try {


            executionResult =

                await executeExecutionStep(

                    context

                );


        } catch(error) {


            executionResult = {


                success:false,


                failure:

                {

                    stage:
                        "execution",


                    failureType:
                        "execution-error",


                    reason:

                        error?.message ||

                        "Execution exception"

                }

            };


        }








        updateTraceFromResult(

            context.trace,

            executionResult

        );









        /*
         * SUCCESS
         */


        if (
            executionResult?.success === true
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

                executionResult.result,

                context

            );


        }









        /*
         * FAILURE
         */


        lastFailure =

            executionResult?.failure ||

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
         * FAILURE DECISION
         */


        const decision =

            await handleExecutionFailure(

                context,

                lastFailure

            );









        switch(

            decision.action

        ) {


            case FAILURE_ACTION.RETRY:


                addTraceEvent(

                    context.trace,

                    "RETRY"

                );


                continue;









            case FAILURE_ACTION.REPLAN:


                const replanned =

                    await processReplan(

                        context,

                        decision

                    );



                if (
                    replanned
                ) {

                    continue;

                }



                break;









            case FAILURE_ACTION.CLARIFICATION:


            case FAILURE_ACTION.FINISH:


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
         * Если сюда дошли —
         * дальше двигаться нельзя.
         */


        break;


    }









    /*
     * =====================================================
     * LIMIT
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

            lastFailure ||

            {

                stage:
                    "execution",


                failureType:
                    "execution-limit",


                reason:
                    "Execution limit reached"

            }

        ),

        context

    );


}
