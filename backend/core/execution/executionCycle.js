/*
 * =========================================================
 * JESSICA EXECUTION CYCLE v12
 * =========================================================
 *
 * Central Execution Coordinator.
 *
 *
 * Ответственность:
 *
 * - создать Execution Context;
 * - создать Execution Trace;
 * - запустить Execution Loop;
 * - передать результат в Terminal.
 *
 *
 * Flow:
 *
 * Task
 *   ↓
 * Context Factory
 *   ↓
 * Trace
 *   ↓
 * Execution Loop
 *   ↓
 * Trace Sync
 *   ↓
 * Terminal Result
 *
 *
 * НЕ:
 *
 * - выполняет Tools;
 * - делает Retry;
 * - делает Replan;
 * - анализирует ошибки;
 * - создаёт Answer;
 * - валидирует результат.
 *
 * =========================================================
 */



import {
    createExecutionContext
} from "./executionContext.js";



import {
    createExecutionTrace,
    addTraceEvent
} from "../trace/executionTrace.js";



import {
    syncTraceFromContext
} from "../trace/traceContextAdapter.js";



import {
    executeExecutionLoop
} from "./cycle/executionLoop.js";



import {
    finishFailedExecution,
    finishSuccessfulExecution
} from "./cycle/cycleTerminal.js";









/*
 * =========================================================
 * SAFE FAILURE
 * =========================================================
 */


function buildExecutionException(

    error

) {


    return {


        stage:

            "execution",



        failureType:

            "execution-exception",



        reason:

            error?.message ||

            "Execution exception"


    };

}









/*
 * =========================================================
 * EXECUTE PLAN CYCLE
 * =========================================================
 */


export async function executePlanCycle(

    task,

    initialPlan,

    planningContext = {}

) {


    /*
     * =====================================================
     * CREATE CONTEXT
     * =====================================================
     */


    const context =

        createExecutionContext({

            task,

            plan:

                initialPlan,


            planningContext

        });









    /*
     * =====================================================
     * CREATE TRACE
     * =====================================================
     */


    context.trace =

        createExecutionTrace(

            task

        );









    /*
     * Первичная синхронизация.
     *
     * Trace получает:
     *
     * - execution metadata;
     * - counters;
     * - initial context state.
     *
     */


    syncTraceFromContext(

        context.trace,

        context

    );









    addTraceEvent(

        context.trace,

        "EXECUTION_STARTED",

        {

            executionId:

                context.executionId

        }

    );









    /*
     * =====================================================
     * EXECUTION LOOP
     * =====================================================
     */


    let result;



    try {


        result =

            await executeExecutionLoop(

                context

            );


    }

    catch(error){


        result = {


            success:false,


            failure:

                buildExecutionException(

                    error

                )


        };


    }









    /*
     * =====================================================
     * FINAL TRACE SYNC
     * =====================================================
     *
     * Перед Terminal переносим:
     *
     * - steps;
     * - failures;
     * - attempts;
     * - replans.
     *
     */


    syncTraceFromContext(

        context.trace,

        context

    );









    /*
     * =====================================================
     * SUCCESS
     * =====================================================
     */


    if (

        result?.success === true

    ) {


        return finishSuccessfulExecution(

            context,

            result.result

        );


    }









    /*
     * =====================================================
     * FAILURE
     * =====================================================
     */


    return finishFailedExecution(

        context,

        result?.failure

        ||

        {

            stage:

                "execution",



            failureType:

                "unknown",



            reason:

                "Unknown execution failure"

        }

    );


}
