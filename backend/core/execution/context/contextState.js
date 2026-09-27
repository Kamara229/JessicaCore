/*
 * =========================================================
 * JESSICA CONTEXT STATE v3
 * =========================================================
 *
 * Управление состоянием Execution Context.
 *
 *
 * Отвечает:
 *
 * - изменение state;
 * - синхронизация status;
 * - завершение Execution Context.
 *
 *
 * НЕ:
 *
 * - создаёт Context;
 * - выполняет шаги;
 * - принимает Execution решения;
 * - управляет Retry;
 * - управляет Replan.
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
 * VALIDATE STATE
 * =========================================================
 */


function isValidState(

    state

) {


    return Object.values(

        EXECUTION_CONTEXT_STATE

    )
    .includes(

        state

    );

}









/*
 * =========================================================
 * STATUS FROM STATE
 * =========================================================
 */


function resolveStatus(

    state

) {


    switch(

        state

    ){


        case EXECUTION_CONTEXT_STATE.COMPLETED:

            return "COMPLETED";



        case EXECUTION_CONTEXT_STATE.FAILED:

            return "FAILED";



        case EXECUTION_CONTEXT_STATE.FINISHED:

            return "FINISHED";



        case EXECUTION_CONTEXT_STATE.RUNNING:

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


        return null;

    }









    /*
     * Неизвестный state не должен
     * неожиданно переводить Context
     * обратно в RUNNING.
     */


    if(

        !isValidState(

            state

        )

    ){


        return context;

    }









    context.state =

        state;



    context.status =

        resolveStatus(

            state

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


        return null;

    }









    /*
     * Для finish допускаются только
     * терминальные состояния.
     *
     * RUNNING здесь не должен
     * завершать Context.
     */


    const terminalState =

        state === EXECUTION_CONTEXT_STATE.COMPLETED

        ||

        state === EXECUTION_CONTEXT_STATE.FAILED

        ||

        state === EXECUTION_CONTEXT_STATE.FINISHED

            ?

            state

            :

            EXECUTION_CONTEXT_STATE.FINISHED;









    updateExecutionState(

        context,

        terminalState

    );









    context.finishedAt =

        new Date()

            .toISOString();









    return context;

}
