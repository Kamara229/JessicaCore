/*
 * =========================================================
 * JESSICA TRACE LIFECYCLE v1
 * =========================================================
 *
 * Lifecycle Execution Trace.
 *
 *
 * Ответственность:
 *
 * - завершить Trace;
 * - определить финальный Trace Status;
 * - отметить готовность к Learning.
 *
 *
 * НЕ:
 *
 * - создаёт Result;
 * - записывает события Execution;
 * - принимает Failure Decision.
 *
 * =========================================================
 */



/*
 * =========================================================
 * FINISH
 * =========================================================
 */


export function finishExecutionTrace(

    trace

){


    if(!trace)
        return trace;



    trace.finishedAt =

        new Date()
        .toISOString();



    switch(
        trace.result?.status
    ){


        case "COMPLETED":


            trace.status =

                "COMPLETED";


            break;



        case "NO_VERIFIED_RESULT":


            trace.status =

                "NO_VERIFIED_RESULT";


            break;



        case "NEEDS_CLARIFICATION":


            trace.status =

                "NEEDS_CLARIFICATION";


            break;



        default:


            trace.status =

                "FAILED";

    }



    trace.learningReady =

        true;



    return trace;

}
