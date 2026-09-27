/*
 * =========================================================
 * JESSICA CONTEXT STATE v2
 * =========================================================
 *
 * Управление состоянием Execution Context.
 *
 *
 * Отвечает:
 *
 * - изменение state;
 * - синхронизация status;
 * - завершение execution.
 *
 *
 * НЕ:
 *
 * - создаёт Context;
 * - выполняет шаги;
 * - принимает решения;
 * - управляет Retry/Replan.
 *
 * =========================================================
 */







/*
 * =========================================================
 * STATES
 * =========================================================
 */


export const EXECUTION_CONTEXT_STATE = {


    RUNNING:

        "RUNNING",



    COMPLETED:

        "COMPLETED",



    FAILED:

        "FAILED",



    FINISHED:

        "FINISHED"


};









/*
 * =========================================================
 * NORMALIZE STATE
 * =========================================================
 */


function normalizeState(

    state

) {


    if (

        Object.values(

            EXECUTION_CONTEXT_STATE

        )
        .includes(state)

    ) {


        return state;

    }



    return EXECUTION_CONTEXT_STATE.RUNNING;

}









/*
 * =========================================================
 * STATUS FROM STATE
 * =========================================================
 */


function resolveStatus(

    state

) {


    switch(state){


        case EXECUTION_CONTEXT_STATE.COMPLETED:

            return "COMPLETED";



        case EXECUTION_CONTEXT_STATE.FAILED:

            return "FAILED";



        case EXECUTION_CONTEXT_STATE.FINISHED:

            return "FINISHED";



        default:

            return "ACTIVE";

    }

}









/*
 * =========================================================
 * UPDATE STATE
 * =========================================================
 */


export function updateExecutionState(

    context,

    state

) {


    if(

        !context

    ){

        return context;

    }









    const normalized =

        normalizeState(

            state

        );









    context.state =

        normalized;



    context.status =

        resolveStatus(

            normalized

        );









    return context;

}









/*
 * =========================================================
 * FINISH CONTEXT
 * =========================================================
 */


export function finishExecutionContext(

    context,

    state = EXECUTION_CONTEXT_STATE.FINISHED

) {


    if(

        !context

    ){

        return context;

    }









    updateExecutionState(

        context,

        state

    );









    context.finishedAt =

        new Date()

            .toISOString();









    return context;

}
