/*
 * =========================================================
 * JESSICA CONTEXT STATE v1
 * =========================================================
 *
 * Управление состоянием Execution Context.
 *
 *
 * Отвечает:
 *
 * - изменение state;
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
 * UPDATE STATE
 * =========================================================
 */


export function updateExecutionState(

    context,

    state

) {


    if (

        !context

    ) {

        return context;

    }







    context.state =

        state;







    return context;

}









/*
 * =========================================================
 * FINISH CONTEXT
 * =========================================================
 */


export function finishExecutionContext(

    context,

    state = "FINISHED"

) {


    if (

        !context

    ) {

        return;

    }







    context.state =

        state;







    context.status =



        state === "COMPLETED"

            ?

            "COMPLETED"



            :



        state === "FAILED"

            ?

            "FAILED"



            :



            "FINISHED";







    context.finishedAt =

        new Date()

            .toISOString();

}
