/*
 * =========================================================
 * JESSICA CONTEXT REPLAN v2
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
 * SAFE NUMBER
 * =========================================================
 */


function safeNumber(

    value

) {


    const number =

        Number(value);



    return Number.isFinite(number)

        ?

        number

        :

        0;

}









/*
 * =========================================================
 * RESET EXECUTION AFTER REPLAN
 * =========================================================
 */


export function resetExecutionAfterReplan(

    context

) {


    if(

        !context

    ){

        return context;

    }









    context.attempt =

        0;



    context.retryCount =

        0;









    /*
     * Очистка результатов
     * старого execution pass
     */


    context.runResult =

        null;



    context.answerResult =

        null;



    context.validationResult =

        null;



    context.terminalResult =

        null;









    return context;

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


    if(

        !context

    ){

        return null;

    }









    context.replanCount =

        safeNumber(

            context.replanCount

        )

        +

        1;









    const record = {


        replanCount:

            context.replanCount,



        previousPlan:

            data.previousPlan ||

            null,



        newPlan:

            data.newPlan ||

            null,



        failure:

            data.failure ||

            null,



        timestamp:

            new Date()

                .toISOString()


    };









    if(

        !Array.isArray(

            context.replanHistory

        )

    ){

        context.replanHistory = [];

    }









    context.replanHistory.push(

        record

    );









    /*
     * Legacy compatibility
     */


    if(

        !Array.isArray(

            context.replans

        )

    ){

        context.replans = [];

    }









    context.replans.push(

        record

    );









    return record;

}
