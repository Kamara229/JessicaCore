/*
 * =========================================================
 * JESSICA TASK RUNNER
 * ARGUMENT RESOLVER v3
 * =========================================================
 *
 * Главный маршрутизатор подготовки arguments.
 *
 *
 * Flow:
 *
 * Step Arguments
 *        ↓
 * Route Resolver
 *        ↓
 * Final Arguments
 *
 *
 * Resolver:
 *
 * web_fetch
 *      ↓
 * fetchSourceResolver
 *
 *
 * остальные tools
 *      ↓
 * valueResolver
 *
 *
 * НЕ:
 *
 * - выполняет Tools;
 * - читает Execution Loop;
 * - делает Retry;
 * - делает Replan.
 *
 * =========================================================
 */


import {
    resolveValue
} from "./argumentResolver/valueResolver.js";


import {
    resolveFetchSource
} from "./argumentResolver/fetchSourceResolver.js";









/*
 * =========================================================
 * WEB FETCH ARGUMENTS
 * =========================================================
 */


async function resolveFetchArguments(

    originalArgs,

    results,

    selectionContext

) {


    const source =

        await resolveFetchSource(

            originalArgs,

            results,

            selectionContext

        );









    /*
     * Resolver не вмешивается.
     *
     * Используем обычный путь.
     */


    if (

        source === null

    ) {


        return resolveValue(

            originalArgs || {},

            results

        );

    }









    /*
     * Ошибка выбора источника.
     */


    if (

        source.success !== true

    ) {


        return source;

    }









    const argsWithoutUrl = {


        ...(originalArgs || {})


    };







    delete argsWithoutUrl.url;








    const resolved =

        resolveValue(

            argsWithoutUrl,

            results

        );









    if (

        resolved.success !== true

    ) {


        return resolved;

    }









    return {


        success:true,


        value:

        {


            ...resolved.value,



            url:

                source.url


        }


    };

}









/*
 * =========================================================
 * MAIN RESOLVER
 * =========================================================
 */


export async function resolveStepArguments(

    toolName,

    originalArgs = {},

    results = [],

    selectionContext = ""

) {



    /*
     * =====================================================
     * WEB FETCH ROUTE
     * =====================================================
     */


    if (

        toolName === "web_fetch"

    ) {


        return resolveFetchArguments(

            originalArgs,

            results,

            selectionContext

        );

    }









    /*
     * =====================================================
     * STANDARD ROUTE
     * =====================================================
     */


    return resolveValue(

        originalArgs,

        results

    );

}
