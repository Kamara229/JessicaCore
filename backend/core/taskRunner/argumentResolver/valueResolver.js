/*
 * =========================================================
 * JESSICA TASK RUNNER
 * VALUE RESOLVER v1
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









/*
 * =========================================================
 * RESOLVE VALUE
 * =========================================================
 */


export function resolveValue(

    value,

    results

) {



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

                    results

                );







            if (

                !resolved.success

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




        const resolved =

            resolveValue(

                item,

                results

            );







        if (

            !resolved.success

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
