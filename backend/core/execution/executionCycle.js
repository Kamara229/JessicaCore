/*
 * =========================================================
 * JESSICA EXECUTION CYCLE v10
 * =========================================================
 *
 * Central Execution Coordinator.
 *
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
 * - сохраняет Learning.
 *
 * =========================================================
 */


import {
    createExecutionContext,
    registerExecutionFailure,
    registerExecutionAttempt,
    finishExecutionContext
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


            ...(result?.executionMeta || {}),



            executionId:

                context?.executionId || null,



            traceId:

                context?.trace?.id || null,



            attempt:

                context?.attempt || 0,



            retryCount:

                context?.retryCount || 0,



            replanCount:

                context?.replanCount || 0


        }


    };


}









/*
 * =========================================================
 * SAFE FAILURE
 * =========================================================
 */


function normalizeFailure(

    failure

) {


    return {


        stage:

            failure?.stage ||

            "execution",



        failureType:

            failure?.failureType ||

            "execution-error",



        reason:

            failure?.reason ||

            "Неизвестная ошибка"


    };

}









/*
 * =========================================================
 * REPLAN
 * =========================================================
 */


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








    const previousPlan =

        context.plan;









    const alternative =

        await createAlternativePlan(

            context,

            decision.failure

        );









    if (

        !alternative?.success

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









    context.attempt =
        0;



    context.retryCount =
        0;









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








    addTraceEvent(

        context.trace,

        "REPLAN_COMPLETED"

    );








    return true;

}









/*
 * =========================================================
 * MAIN
 * =========================================================
 */


export async function executePlanCycle(

    task,

    initialPlan,

    planningContext = {}

) {



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

        "EXECUTION_STARTED",

        {

            executionId:

                context.executionId

        }

    );









    let lastFailure =
        null;









    while (

        true

    ) {






        if (

            context.attempt >=

            MAX_EXECUTION_ATTEMPTS

        ) {


            break;

        }









        registerExecutionAttempt(

            context,

            {

                retryCount:

                    context.retryCount,



                replanCount:

                    context.replanCount

            }

        );









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









        try {


            result =

                await executeExecutionStep(

                    context

                );



        } catch(error) {



            result = {


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

            result

        );









        /*
         * =================================================
         * SUCCESS
         * =================================================
         */


        if (

            result?.success === true

        ) {


            finishExecutionContext(

                context,

                "COMPLETED"

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
         * =================================================
         * FAILURE
         * =================================================
         */


        lastFailure =

            normalizeFailure(

                result?.failure

            );









        registerExecutionFailure(

            context,

            lastFailure

        );









        const decision =

            await handleExecutionFailure(

                context,

                lastFailure

            );









        switch (

            decision.action

        ) {







            case FAILURE_ACTION.RETRY:


                context.retryCount++;


                addTraceEvent(

                    context.trace,

                    "RETRY",

                    {

                        attempt:

                            context.attempt

                    }

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


                finishExecutionContext(

                    context,

                    "FAILED"

                );



                finishExecutionTrace(

                    context.trace

                );



                return attachExecutionMeta(

                    buildTerminalResult(

                        context,

                        {

                            ...lastFailure,


                            needsClarification:

                                true


                        }

                    ),

                    context

                );









            case FAILURE_ACTION.FINISH:


                break;


        }



        break;


    }









    /*
     * =====================================================
     * TERMINAL
     * =====================================================
     */


    finishExecutionContext(

        context,

        "FAILED"

    );



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
