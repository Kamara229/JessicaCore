/*
 * =========================================================
 * JESSICA EXECUTION CYCLE v9
 * =========================================================
 *
 * Central Execution Coordinator.
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
    addTraceEvent,
    addTraceAttempt,
    addTraceReplan
} from "../trace/executionTrace.js";


import {
    MAX_EXECUTION_ATTEMPTS
} from "./retryPolicy.js";









function attachExecutionMeta(

    result,

    context

){

    return {

        ...result,


        executionMeta:{

            ...(result?.executionMeta || {}),


            executionId:
                context.executionId,


            traceId:
                context.trace?.id || null,


            attempt:
                context.attempt,


            retryCount:
                context.retryCount || 0,


            replanCount:
                context.replanCount || 0


        }

    };

}









function registerFailure(

    context,

    failure

){

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

){

    addTraceEvent(

        context.trace,

        "REPLAN_STARTED",

        decision.failure

    );



    const previousPlan =
        context.plan;



    const alternative =

        await createAlternativePlan(

            context,

            decision.failure

        );



    if(
        !alternative.success
    ){

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



    if(!applied){

        return false;

    }





    /*
     * Новый маршрут =
     * новый цикл попыток
     */

    context.attempt = 0;



    context.retryCount = 0;



    addTraceReplan(

        context.trace,

        {

            previousPlan,


            newPlan:
                context.plan,


            failure:
                decision.failure

        }

    );



    return true;

}









export async function executePlanCycle(

    task,

    initialPlan,

    planningContext = {}

){



    const context =

        createExecutionContext({

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

        "EXECUTION_STARTED"

    );






    let lastFailure = null;








    while(true){



        if(
            context.attempt >=
            MAX_EXECUTION_ATTEMPTS
        ){

            break;

        }





        context.attempt++;



        addTraceAttempt(

            context.trace,

            context.attempt,

            {

                retryCount:
                    context.retryCount,


                replanCount:
                    context.replanCount

            }

        );







        let result;



        try{


            result =

                await executeExecutionStep(

                    context

                );


        }

        catch(error){


            result={

                success:false,


                failure:{

                    stage:
                        "execution",


                    failureType:
                        "execution-error",


                    reason:
                        error?.message ||

                        "Execution error"

                }

            };

        }








        updateTraceFromResult(

            context.trace,

            result

        );








        if(
            result?.success === true
        ){


            finishExecutionTrace(

                context.trace

            );


            return attachExecutionMeta(

                result.result,

                context

            );


        }








        lastFailure =

            result?.failure ||

            {

                failureType:
                    "unknown",


                reason:
                    "Unknown failure"

            };






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
        ){


            case FAILURE_ACTION.RETRY:


                context.retryCount++;


                addTraceEvent(

                    context.trace,

                    "RETRY"

                );


                continue;







            case FAILURE_ACTION.REPLAN:


                if(

                    await executeReplan(

                        context,

                        decision

                    )

                ){

                    continue;

                }


                break;








            case FAILURE_ACTION.CLARIFICATION:


                return attachExecutionMeta(

                    buildTerminalResult(

                        context,

                        {

                            ...lastFailure,

                            needsClarification:true

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

            lastFailure ||

            {

                failureType:
                    "execution-limit",


                reason:
                    "Execution limit reached"

            }

        ),

        context

    );

}
