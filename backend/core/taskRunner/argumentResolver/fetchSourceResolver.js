/*
 * =========================================================
 * JESSICA TASK RUNNER
 * FETCH SOURCE RESOLVER v2
 * =========================================================
 *
 * Выбор источника для web_fetch.
 *
 *
 * Flow:
 *
 * web_search result
 *        ↓
 * Search Candidates
 *        ↓
 * Source Selector
 *        ↓
 * Selected URL
 *
 *
 * НЕ:
 *
 * - выполняет web_fetch;
 * - выполняет Tools;
 * - делает Retry;
 * - делает Replan;
 * - меняет Execution Context.
 *
 * =========================================================
 */


import {
    selectSource
} from "../../sourceSelector.js";


import {
    findPreviousSearchResult,
    extractSearchCandidates
} from "./stepResultResolver.js";









/*
 * =========================================================
 * FAILURE
 * =========================================================
 */


function buildFailure({

    failureType,

    text,

    shouldRetry = false,

    needsClarification = false

}) {


    return {


        success:

            false,



        stage:

            "source-selection",



        failureType,



        text,



        reason:

            text,



        shouldRetry:

            shouldRetry === true,



        needsClarification:

            needsClarification === true


    };

}









/*
 * =========================================================
 * DIRECT URL CHECK
 * =========================================================
 */


function isDirectUrl(

    url

) {


    return (

        typeof url === "string"

        &&

        /^https?:\/\/\S+$/i.test(

            url.trim()

        )

    );

}









/*
 * =========================================================
 * RESOLVE FETCH SOURCE
 * =========================================================
 */


export async function resolveFetchSource(

    originalArgs,

    results,

    selectionContext

) {


    /*
     * =====================================================
     * DIRECT URL
     * =====================================================
     */


    if (

        isDirectUrl(

            originalArgs?.url

        )

    ) {


        return {


            success:

                true,



            url:

                originalArgs.url.trim()


        };


    }









    /*
     * =====================================================
     * FIND SEARCH RESULT
     * =====================================================
     */


    const searchResult =

        findPreviousSearchResult(

            results

        );








    /*
     * Нет предыдущего поиска.
     *
     * Для web_fetch это ошибка маршрута,
     * а не отсутствие аргумента.
     */


    if (

        !searchResult

    ) {


        return buildFailure({

            failureType:

                "missing-fetch-source",



            shouldRetry:

                false,



            text:

                "Для web_fetch не найден результат web_search"

        });


    }









    /*
     * =====================================================
     * EXTRACT CANDIDATES
     * =====================================================
     */


    const candidates =

        extractSearchCandidates(

            searchResult

        );








    if (

        candidates.length === 0

    ) {


        return buildFailure({

            failureType:

                "no-search-results",



            shouldRetry:

                true,



            text:

                "Поиск не вернул ссылок для загрузки"

        });


    }









    /*
     * =====================================================
     * SOURCE SELECTOR
     * =====================================================
     */


    let selection;



    try {


        selection =

            await selectSource(

                selectionContext,

                candidates

            );


    }

    catch(error){


        console.error(

            "Jessica Source Selector error:",

            error

        );



        return buildFailure({

            failureType:

                "source-selector-error",



            text:

                "Ошибка выбора источника"

        });


    }









    /*
     * =====================================================
     * NO SUITABLE SOURCE
     * =====================================================
     */


    if (

        selection?.noSuitableSource === true

    ) {


        return buildFailure({

            failureType:

                "no-suitable-source",



            shouldRetry:

                true,



            text:

                selection.reason ||

                "Не найден подходящий источник"

        });


    }









    /*
     * =====================================================
     * SELECTOR FAILURE
     * =====================================================
     */


    if (

        selection?.success !== true

        ||

        !selection?.result?.url

    ) {


        return buildFailure({

            failureType:

                "source-selector-error",



            text:

                selection?.reason ||

                "Источник не выбран"

        });


    }









    /*
     * =====================================================
     * SUCCESS
     * =====================================================
     */


    return {


        success:

            true,



        url:

            selection.result.url


    };


}
