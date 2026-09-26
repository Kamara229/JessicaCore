/*
 * =========================================================
 * JESSICA EXECUTION CYCLE v7
 * =========================================================
 *
 * Центральный координатор Execution Run.
 *
 * Flow:
 *
 * Plan
 *   ↓
 * Context
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

            planningContext?.experience ||
            null


    };

}









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









function attachExecutionMeta(

    result,

    context

) {


    return {


        ...result,


        executionMeta:

        {


            ...(result?.executionMeta || {}),



            executionId:

                context.executionId,



            traceId:

                context.trace?.id || null,



            attempts:

                context.attempt,



            retryCount:

                context.retryCount || 0,



            replans:

                context.replanCount || 0,



            experience:

            {

                used:

                    context.experience
                        ?.skills
                        ?.length > 0,



                skills:

                    context.experience?.skills || []

            }


        }


    };

}









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









async function executeReplan(

    context,

    decision

) {


    addTraceEvent(

        context.trace,

        "REPLAN_STARTED",

        {

            failure:
                decision.failure

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






    applyAlternativePlan(

        context,

        alternative

    );





    context.replanCount++;


    context.retryCount = 0;





    addTraceEvent(

        context.trace,

        "REPLAN_COMPLETED",

        {

            intent:
                context.plan.intent

        }

    );



    return true;

}









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






    let lastFailure = null;








    while (

        context.attempt <

        MAX_EXECUTION_ATTEMPTS

    ) {



        context.attempt++;





        addTraceEvent(

            context.trace,

            "ATTEMPT_STARTED",

            {

                attempt:
                    context.attempt

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
                        "execution-error",

                    reason:
                        error.message

                }

            };


        }







        updateTraceFromResult(

            context.trace,

            result

        );









        if (
            result.success === true
        ) {


            context.state =
                "COMPLETED";


            finishExecutionTrace(

                context.trace

            );



            return attachExecutionMeta(

                result.result,

                context

            );


        }







        lastFailure =

            result.failure;



        registerFailure(

            context,

            lastFailure

        );







        const decision =

            await handleExecutionFailure(

                context,

                lastFailure

            );








        switch(
            decision.action
        ) {



            case FAILURE_ACTION.RETRY:


                context.retryCount++;


                addTraceEvent(

                    context.trace,

                    "RETRY"

                );


                continue;








            case FAILURE_ACTION.REPLAN:


                if (

                    await executeReplan(

                        context,

                        decision

                    )

                ) {

                    continue;

                }

                break;








            case FAILURE_ACTION.CLARIFICATION:


                finishExecutionTrace(

                    context.trace

                );


                return attachExecutionMeta(

                    buildTerminalResult(

                        context,

                        {

                            status:
                                "NEEDS_CLARIFICATION",

                            ...lastFailure

                        }

                    ),

                    context

                );








            case FAILURE_ACTION.FINISH:

                break;


        }



        break;

    }








    finishExecutionTrace(

        context.trace

    );



    return attachExecutionMeta(

        buildTerminalResult(

            context,

            {

                status:
                    "FAILED",


                ...(

                    lastFailure ||

                    {}

                )

            }

        ),

        context

    );


}
