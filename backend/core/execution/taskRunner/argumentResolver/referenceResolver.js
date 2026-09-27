/*
 * =========================================================
 * JESSICA TASK RUNNER
 * REFERENCE RESOLVER v1
 * =========================================================
 *
 * Разрешение ссылок между Execution Steps.
 *
 *
 * Поддерживает:
 *
 * {
 *    $from:"step_id",
 *    path:"data.value"
 * }
 *
 *
 * Flow:
 *
 * Reference
 *     ↓
 * Find Step Result
 *     ↓
 * Read Path
 *     ↓
 * Value
 *
 *
 * НЕ:
 *
 * - выполняет Tools;
 * - делает Retry;
 * - делает Replan;
 * - меняет Execution Context.
 *
 * =========================================================
 */


import {
    findStepResult
} from "./stepResultResolver.js";









/*
 * =========================================================
 * FAILURE
 * =========================================================
 */


function buildReferenceFailure(

    failureType,

    text

) {


    return {


        success:false,


        stage:
            "argument-resolution",



        failureType,



        text,



        reason:
            text,



        shouldRetry:false


    };

}









/*
 * =========================================================
 * OBJECT PATH
 * =========================================================
 */


function getValueByPath(

    source,

    path

) {


    if (

        source === null ||

        source === undefined

    ) {

        return undefined;

    }





    if (

        typeof path !== "string"

        ||

        !path.trim()

    ) {

        return source;

    }







    const parts =

        path

            .split(".")

            .map(

                item =>

                    item.trim()

            )

            .filter(Boolean);







    let current =

        source;







    for(

        const part

        of parts

    ) {





        if (

            current === null ||

            current === undefined

        ) {

            return undefined;

        }







        /*
         * Защита от prototype traversal
         */


        if (

            part === "__proto__"

            ||

            part === "prototype"

            ||

            part === "constructor"

        ) {

            return undefined;

        }







        current =

            current[part];


    }







    return current;


}









/*
 * =========================================================
 * RESOLVE REFERENCE
 * =========================================================
 */


export function resolveReference(

    reference,

    results

) {


    const from =

        typeof reference?.$from === "string"

            ?

            reference.$from.trim()

            :

            "";








    if (

        !from

    ) {


        return buildReferenceFailure(

            "missing-from",

            "В ссылке отсутствует $from"

        );

    }









    const source =

        findStepResult(

            from,

            results

        );









    if (

        !source

    ) {


        return buildReferenceFailure(

            "previous-step-not-found",

            `Не найден результат шага ${from}`

        );

    }









    if (

        source.success !== true

    ) {


        return buildReferenceFailure(

            "previous-step-failed",

            `Предыдущий шаг ${from} завершился ошибкой`

        );

    }









    const value =

        getValueByPath(

            source,

            reference.path

        );









    if (

        value === undefined

    ) {


        return buildReferenceFailure(

            "reference-path-not-found",

            (
                `Не найден путь ` +

                `${reference.path || "result"} ` +

                `в шаге ${from}`
            )

        );

    }









    return {


        success:true,


        value


    };

}









/*
 * =========================================================
 * CHECK REFERENCE
 * =========================================================
 */


export function isReference(

    value

) {


    return (

        value

        &&

        typeof value === "object"

        &&

        !Array.isArray(value)

        &&

        typeof value.$from === "string"

    );

}
