/*
 * =========================================================
 * JESSICA EXECUTION
 * CYCLE TERMINAL v3
 * =========================================================
 *
 * Финальное завершение Execution Cycle.
 *
 *
 * Flow:
 *
 * Context
 *    ↓
 * Sync Trace
 *    ↓
 * Finish Context
 *    ↓
 * Finish Trace
 *    ↓
 * Result
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


import {
    syncTraceFromContext
} from "../../trace/traceContextAdapter.js";









/*
 * =========================================================
 * TRACE PREPARE
 * =========================================================
 */


function prepareTrace(

    context

) {


    if(

        !context?.trace

    ){

        return;

    }



    syncTraceFromContext(

        context.trace,

        context

    );


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


    prepareTrace(

        context

    );









    finishExecutionContext(

        context,

        "FAILED"

    );









    finishExecutionTrace(

        context.trace

    );









    return buildTerminalResult(

        context,

        failure

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


    prepareTrace(

        context

    );









    finishExecutionContext(

        context,

        "COMPLETED"

    );









    finishExecutionTrace(

        context.trace

    );









    return result;


}
