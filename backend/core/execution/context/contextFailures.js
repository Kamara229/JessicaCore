/*
 * =========================================================
 * JESSICA CONTEXT FAILURES v3
 * =========================================================
 *
 * Хранение Failure в Execution Context.
 *
 *
 * Отвечает:
 *
 * - сохранение последней Failure;
 * - накопление истории Failure;
 * - увеличение failureCount.
 *
 *
 * НЕ:
 *
 * - нормализует Failure;
 * - классифицирует Failure;
 * - принимает решение Retry;
 * - принимает решение Replan;
 * - меняет Execution Flow.
 *
 *
 * Failure должна приходить сюда уже после
 * Failure Normalizer / Failure Classifier.
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
 * ENSURE FAILURE HISTORY
 * =========================================================
 */


function ensureFailureHistory(

    context

) {


    if(

        !Array.isArray(

            context.errors

        )

    ){


        context.errors = [];

    }



    return context.errors;

}









/*
 * =========================================================
 * CREATE FAILURE RECORD
 * =========================================================
 */


function createFailureRecord(

    failure

) {


    if(

        failure &&

        typeof failure === "object"

    ){


        return {


            ...failure,



            timestamp:

                new Date()

                    .toISOString()


        };

    }









    return {


        stage:

            "execution",



        failureType:

            "unknown",



        category:

            "execution",



        reason:

            String(

                failure ||

                "Неизвестная ошибка"

            ),



        timestamp:

            new Date()

                .toISOString()


    };

}









/*
 * =========================================================
 * REGISTER FAILURE
 * =========================================================
 */


export function registerExecutionFailure(

    context,

    failure

) {


    if(

        !context

    ){


        return null;

    }









    /*
     * =====================================================
     * RECORD
     * =====================================================
     */


    const record =

        createFailureRecord(

            failure

        );









    /*
     * =====================================================
     * LAST FAILURE
     * =====================================================
     */


    context.lastFailure =

        record;









    /*
     * =====================================================
     * HISTORY
     * =====================================================
     */


    const history =

        ensureFailureHistory(

            context

        );



    history.push(

        record

    );









    /*
     * =====================================================
     * COUNTER
     * =====================================================
     */


    context.failureCount =

        safeNumber(

            context.failureCount

        )

        +

        1;









    return record;

}
