/*
 * =========================================================
 * JESSICA CONTEXT REPLAN
 * =========================================================
 *
 * Управление состоянием после Replan.
 *
 *
 * Отвечает:
 *
 * - регистрация Replan;
 * - подготовка нового execution pass.
 *
 *
 * НЕ:
 *
 * - принимает решение Replan;
 * - создаёт новый Plan.
 *
 * =========================================================
 */







/*
 * =========================================================
 * RESET EXECUTION AFTER REPLAN
 * =========================================================
 */


export function resetExecutionAfterReplan(

    context

) {


    if (

        !context

    ) {

        return;

    }



    context.attempt = 0;


    context.retryCount = 0;



}









/*
 * =========================================================
 * REGISTER REPLAN
 * =========================================================
 */


export function registerExecutionReplan(

    context,

    data = {}

) {


    if (

        !context

    ) {

        return;

    }





    context.replanCount++;





    const record = {


        ...data,


        timestamp:

            new Date()

                .toISOString()


    };





    if (

        !Array.isArray(

            context.replanHistory

        )

    ) {


        context.replanHistory = [];

    }





    context.replanHistory.push(

        record

    );





    if (

        !Array.isArray(

            context.replans

        )

    ) {


        context.replans = [];

    }





    context.replans.push(

        record

    );


}
