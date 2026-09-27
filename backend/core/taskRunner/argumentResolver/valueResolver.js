/*
 * =========================================================
 * JESSICA TASK RUNNER
 * VALUE RESOLVER v2
 * =========================================================
 *
 * Рекурсивное разрешение arguments.
 *
 *
 * Поддерживает:
 *
 * - primitive values;
 * - arrays;
 * - objects;
 * - $from references.
 *
 *
 * Flow:
 *
 * Arguments
 *      ↓
 * Object Traversal
 *      ↓
 * Reference Resolver
 *      ↓
 * Resolved Arguments
 *
 *
 * НЕ:
 *
 * - выполняет Tools;
 * - выбирает источники;
 * - делает Retry;
 * - делает Replan.
 *
 * =========================================================
 */


import {
    resolveReference,
    isReference
} from "./referenceResolver.js";









const MAX_DEPTH = 20;









/*
 * =========================================================
 * UNSAFE KEY
 * =========================================================
 */


function isUnsafeKey(

    key

) {


    return (

        key === "__proto__"

        ||

        key === "prototype"

        ||

        key === "constructor"

    );

}









/*
 * =========================================================
 * FAILURE
 * =========================================================
 */


function buildFailure(

    text

) {


    return {


        success:false,


        stage:

            "argument-resolution",



        failureType:

            "value-resolution-error",



        reason:

            text,



        text,


        shouldRetry:false


    };

}









/*
 * =========================================================
 * RESOLVE VALUE
 * =========================================================
 */


export function resolveValue(

    value,

    results,

    depth = 0

) {


    /*
     * Защита глубины
     */


    if (

        depth >

        MAX_DEPTH

    ) {


        return buildFailure(

            "Превышена максимальная глубина arguments"

        );

    }









    /*
     * =====================================================
     * SIMPLE VALUE
     * =====================================================
     */


    if (

        value === null

        ||

        value === undefined

        ||

        typeof value !== "object"

    ) {


        return {


            success:true,


            value


        };

    }









    /*
     * =====================================================
     * REFERENCE
     * =====================================================
     */


    if (

        isReference(value)

    ) {


        return resolveReference(

            value,

            results

        );

    }









    /*
     * =====================================================
     * ARRAY
     * =====================================================
     */


    if (

        Array.isArray(value)

    ) {


        const output = [];







        for (

            const item

            of value

        ) {


            const resolved =

                resolveValue(

                    item,

                    results,

                    depth + 1

                );







            if (

                resolved.success !== true

            ) {


                return resolved;

            }







            output.push(

                resolved.value

            );


        }







        return {


            success:true,


            value:output


        };


    }









    /*
     * =====================================================
     * OBJECT
     * =====================================================
     */


    const output = {};







    for (

        const [

            key,

            item

        ]

        of Object.entries(value)

    ) {


        if (

            isUnsafeKey(key)

        ) {


            continue;

        }







        const resolved =

            resolveValue(

                item,

                results,

                depth + 1

            );







        if (

            resolved.success !== true

        ) {


            return resolved;

        }







        output[key] =

            resolved.value;


    }








    return {


        success:true,


        value:output


    };

}
