/*
 * =========================================================
 * JESSICA EXECUTION CYCLE v14
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
 * - передать результат в Cycle Terminal.
 *
 *
 * Flow:
 *
 * Task
 *   ↓
 * Context
 *   ↓
 * Trace
 *   ↓
 * Execution Loop
 *   ↓
 * Cycle Terminal
 *   ↓
 * Result
 *
 *
 * НЕ:
 *
 * - выполняет Tools;
 * - делает Retry;
 * - делает Replan;
 * - анализирует Failure;
 * - создаёт Answer;
 * - валидирует Result;
 * - завершает Context;
 * - завершает Trace.
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
    executeExecutionLoop
} from "./cycle/executionLoop.js";


import {
    finishFailedExecution,
    finishSuccessfulExecution
} from "./cycle/cycleTerminal.js";









/*
 * =========================================================
 * EXECUTION EXCEPTION
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

            "Execution exception",



        shouldRetry:

            false,



        needsClarification:

            false,



        noVerifiedResult:

            false


    };

}









/*
 * =========================================================
 * UNKNOWN FAILURE
 * =========================================================
 */


function buildUnknownFailure() {


    return {


        stage:

            "execution",



        failureType:

            "unknown",



        reason:

            "Unknown execution failure",



        shouldRetry:

            false,



        needsClarification:

            false,



        noVerifiedResult:

            false


    };

}









/*
 * =========================================================
 * CREATE TRACE
 * =========================================================
 */


function initializeTrace(

    context

) {


    const trace =

        createExecutionTrace(

            context?.task || ""

        );



    context.trace =

        trace;









    addTraceEvent(

        trace,

        "EXECUTION_STARTED",

        {

            executionId:

                context.executionId,



            hasPlan:

                Boolean(

                    context.plan

                ),



            experienceFound:

                context?.experience?.found === true


        }

    );









    return trace;

}









/*
 * =========================================================
 * EXECUTE LOOP
 * =========================================================
 */


async function runExecutionLoop(

    context

) {


    try {


        return await executeExecutionLoop(

            context

        );


    }

    catch(error){


        const failure =

            buildExecutionException(

                error

            );









        addTraceEvent(

            context.trace,

            "EXECUTION_EXCEPTION",

            {

                failureType:

                    failure.failureType,



                reason:

                    failure.reason


            }

        );









        return {


            success:false,


            failure


        };


    }


}









/*
 * =========================================================
 * EXECUTE PLAN CYCLE
 * =========================================================
 */


export async function executePlanCycle(

    task,

    initialPlan,

    planningContext = {},

    experience = null

) {


    /*
     * =====================================================
     * CONTEXT
     * =====================================================
     */


    const context =

        createExecutionContext({

            task,

            plan:

                initialPlan,


            planningContext,


            experience

        });









    /*
     * =====================================================
     * TRACE
     * =====================================================
     */


    initializeTrace(

        context

    );









    /*
     * =====================================================
     * EXECUTION LOOP
     * =====================================================
     */


    const loopResult =

        await runExecutionLoop(

            context

        );









    /*
     * =====================================================
     * LOOP COMPLETED
     * =====================================================
     */


    addTraceEvent(

        context.trace,

        "EXECUTION_LOOP_COMPLETED",

        {

            success:

                loopResult?.success === true,



            failureType:

                loopResult?.failure?.failureType ||

                null


        }

    );









    /*
     * =====================================================
     * SUCCESS
     * =====================================================
     */


    if (

        loopResult?.success === true

        &&

        loopResult.result

    ) {


        return finishSuccessfulExecution(

            context,

            loopResult.result

        );


    }









    /*
     * =====================================================
     * FAILURE
     * =====================================================
     */


    return finishFailedExecution(

        context,

        loopResult?.failure ||

        buildUnknownFailure()

    );


}
