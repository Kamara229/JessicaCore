/*
 * =========================================================
 * JESSICA TASK RUNNER
 * ARGUMENT RESOLVER v2
 * =========================================================
 *
 * Главный маршрутизатор подготовки arguments.
 *
 *
 * Flow:
 *
 * Step Arguments
 *        ↓
 * Fetch Source Resolver
 *        ↓
 * Value Resolver
 *        ↓
 * Final Arguments
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
 * RESOLVE STEP ARGUMENTS
 * =========================================================
 */


export async function resolveStepArguments(

    toolName,

    originalArgs,

    results,

    selectionContext

) {



    /*
     * =====================================================
     * WEB FETCH SPECIAL ROUTE
     * =====================================================
     */


    if (

        toolName === "web_fetch"

    ) {



        const source =

            await resolveFetchSource(

                originalArgs,

                results,

                selectionContext

            );








        /*
         * Source Resolver ничего
         * не обнаружил.
         *
         * Продолжаем обычный resolve.
         */


        if (

            source === null

        ) {


            return resolveValue(

                originalArgs,

                results

            );

        }








        /*
         * Source Resolver нашёл ошибку.
         */


        if (

            source.success !== true

        ) {


            return source;

        }








        /*
         * URL выбран.
         *
         * Остальные аргументы
         * разрешаем стандартно.
         */


        const argsWithoutUrl = {


            ...(originalArgs || {})


        };





        delete argsWithoutUrl.url;








        const rest =

            resolveValue(

                argsWithoutUrl,

                results

            );








        if (

            rest.success !== true

        ) {


            return rest;

        }








        return {


            success:true,


            value:

            {


                ...rest.value,



                url:

                    source.url


            }


        };


    }









    /*
     * =====================================================
     * STANDARD ARGUMENT RESOLUTION
     * =====================================================
     */


    return resolveValue(

        originalArgs,

        results

    );

}
