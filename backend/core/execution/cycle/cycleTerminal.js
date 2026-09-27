/*
 * =========================================================
 * JESSICA EXECUTION
 * CYCLE TERMINAL v1
 * =========================================================
 *
 * Финальное завершение Execution Cycle.
 *
 *
 * Flow:
 *
 * Context
 *    ↓
 * Finish Context
 *    ↓
 * Finish Trace
 *    ↓
 * Terminal Result
 *
 *
 * НЕ:
 *
 * - выполняет Execution;
 * - делает Retry;
 * - делает Replan.
 *
 * =========================================================
 */


import {
    finishExecutionContext
} from "../executionContext.js";


import {
    buildTerminalResult
} from "../executionTerminal.js";


import {
    finishExecutionTrace
} from "../../trace/executionTrace.js";









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
 * FINISH FAILED
 * =========================================================
 */


export function finishFailedExecution(

    context,

    failure

) {


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

            failure

        ),

        context

    );


}









/*
 * =========================================================
 * FINISH SUCCESS
 * =========================================================
 */


export function finishSuccessfulExecution(

    context,

    result

) {


    finishExecutionContext(

        context,

        "COMPLETED"

    );



    finishExecutionTrace(

        context.trace

    );



    return attachExecutionMeta(

        result,

        context

    );


}
