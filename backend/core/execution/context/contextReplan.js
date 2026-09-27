/*
 * =========================================================
 * JESSICA CONTEXT REPLAN v3
 * =========================================================
 *
 * Управление Replan-состоянием Execution Context.
 *
 *
 * Отвечает:
 *
 * - регистрация успешного Replan;
 * - увеличение replanCount;
 * - хранение канонической replanHistory;
 * - сброс counters нового execution pass.
 *
 *
 * НЕ:
 *
 * - принимает решение Replan;
 * - создаёт новый Plan;
 * - применяет новый Plan;
 * - очищает результаты Plan;
 * - выполняет Execution.
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

        Number(

            value

        );



    return Number.isFinite(

        number

    )

        ?

        number

        :

        0;

}









/*
 * =========================================================
 * ENSURE HISTORY
 * =========================================================
 */


function ensureReplanHistory(

    context

) {


    if(

        !Array.isArray(

            context.replanHistory

        )

    ){


        context.replanHistory = [];

    }



    return context.replanHistory;

}









/*
 * =========================================================
 * RESET EXECUTION PASS
 * =========================================================
 */


export function resetExecutionAfterReplan(

    context

) {


    if(

        !context

    ){


        return null;

    }









    /*
     * Новый Plan начинает собственный
     * execution pass.
     *
     * Сбрасываются только локальные counters:
     *
     * attempt
     * retryCount
     *
     * replanCount является глобальным
     * счётчиком текущего Execution и
     * НЕ сбрасывается.
     *
     * Результаты здесь не очищаются.
     * Это ответственность applyAlternativePlan().
     */



    context.attempt =

        0;



    context.retryCount =

        0;









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









    /*
     * =====================================================
     * COUNTER
     * =====================================================
     */


    context.replanCount =

        safeNumber(

            context.replanCount

        )

        +

        1;









    /*
     * =====================================================
     * RECORD
     * =====================================================
     */


    const record = {


        previousPlan:

            data.previousPlan ||

            null,



        newPlan:

            data.newPlan ||

            null,



        failure:

            data.failure ||

            null,



        replanCount:

            context.replanCount,



        timestamp:

            new Date()

                .toISOString()


    };









    /*
     * =====================================================
     * HISTORY
     * =====================================================
     *
     * replanHistory является единственным
     * каноническим журналом Replan.
     *
     * =====================================================
     */


    const history =

        ensureReplanHistory(

            context

        );



    history.push(

        record

    );









    return record;

}
