/*
 * =========================================================
 * JESSICA TASK RUNNER
 * FETCH SOURCE RESOLVER v1
 * =========================================================
 *
 * Выбор источника для web_fetch.
 *
 *
 * Flow:
 *
 * web_search result
 *        ↓
 * Candidates
 *        ↓
 * Source Selector
 *        ↓
 * URL
 *
 *
 * НЕ:
 *
 * - выполняет web_fetch;
 * - выполняет Tools;
 * - делает Retry;
 * - делает Replan.
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


function buildFailure(

    {

        failureType,

        text,

        shouldRetry = false

    }

) {


    return {


        success:false,


        stage:
            "source-selection",



        failureType,



        text,



        reason:
            text,



        shouldRetry:
            shouldRetry === true


    };

}









/*
 * =========================================================
 * DIRECT URL
 * =========================================================
 */


function isDirectUrl(

    url

) {


    return (

        typeof url === "string"

        &&

        /^https?:\/\//i.test(
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
     * Уже передан URL
     * =====================================================
     */


    if (

        isDirectUrl(
            originalArgs?.url
        )

    ) {


        return {


            success:true,


            url:

                originalArgs.url.trim()


        };


    }









    /*
     * =====================================================
     * Ищем предыдущий web_search
     * =====================================================
     */


    const searchResult =

        findPreviousSearchResult(

            results

        );






    if (

        !searchResult

    ) {


        return null;

    }









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


            shouldRetry:true,


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
     * NO SOURCE
     * =====================================================
     */


    if (

        selection?.noSuitableSource === true

    ) {


        return buildFailure({

            failureType:
                "no-suitable-source",


            shouldRetry:true,


            text:

                selection.reason ||

                "Не найден подходящий источник"


        });


    }









    /*
     * =====================================================
     * SUCCESS
     * =====================================================
     */


    if (

        selection?.result?.url

    ) {


        return {


            success:true,


            url:

                selection.result.url


        };


    }









    return buildFailure({

        failureType:
            "source-selector-error",


        text:

            selection?.reason ||

            "Источник не выбран"


    });


}
