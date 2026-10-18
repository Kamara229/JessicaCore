/*
 * =========================================================
 * JESSICA TASK RUNNER
 * FETCH SOURCE RESOLVER v3
 * =========================================================
 *
 * web_fetch source resolution.
 *
 *
 * Защита:
 *
 * - direct rejected URL cannot be reused;
 * - dynamic Source Selector получает
 *   excludedSourceUrls.
 *
 * =========================================================
 */


import {
    selectSource
} from "../../sourceSelector.js";


import {
    extractSearchCandidates
} from "./stepResultResolver.js";

import {
    resolveReferencedSearchResult,
    scopeCandidatesToSearchQuery,
    selectionContextForSearch
} from "./sourceQueryScope.js";


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
 * DIRECT URL
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
 * NORMALIZE URL
 * =========================================================
 */


function normalizeUrl(
    value
) {

    const text =

        String(
            value || ""
        )
            .trim();


    if(
        !text
    ){

        return "";

    }


    try {


        const url =
            new URL(
                text
            );


        url.hash =
            "";


        url.protocol =
            url.protocol.toLowerCase();


        url.hostname =
            url.hostname.toLowerCase();


        if(
            url.pathname.length > 1
            &&
            url.pathname.endsWith("/")
        ){

            url.pathname =
                url.pathname.slice(
                    0,
                    -1
                );

        }


        return url.toString();


    }catch(error){


        return text;

    }

}


/*
 * =========================================================
 * EXCLUDED URLS
 * =========================================================
 */


function getExcludedUrls(
    selectionContext
) {

    if(
        !selectionContext
        ||
        typeof selectionContext !== "object"
        ||
        Array.isArray(
            selectionContext
        )
    ){

        return new Set();
    }


    const values =

        Array.isArray(
            selectionContext.excludedSourceUrls
        )

            ? selectionContext.excludedSourceUrls

            : [];


    return new Set(

        values

            .map(
                normalizeUrl
            )

            .filter(Boolean)

    );

}


/*
 * =========================================================
 * RESOLVE
 * =========================================================
 */


export async function resolveFetchSource(

    originalArgs,

    results,

    selectionContext

) {

    const excludedUrls =

        getExcludedUrls(
            selectionContext
        );


    /*
     * =====================================================
     * DIRECT URL
     * =====================================================
     */


    if(
        isDirectUrl(
            originalArgs?.url
        )
    ){

        const directUrl =

            originalArgs.url.trim();


        if(
            excludedUrls.has(
                normalizeUrl(
                    directUrl
                )
            )
        ){

            return buildFailure({

                failureType:
                    "rejected-source-reuse",

                shouldRetry:
                    true,

                text:

                    (
                        "Execution Plan попытался повторно использовать "
                        +
                        `уже отклонённый источник: ${directUrl}`
                    )

            });

        }


        return {

            success:
                true,

            url:
                directUrl

        };

    }


    /*
     * =====================================================
     * PREVIOUS SEARCH
     * =====================================================
     */


    const searchResult =

        resolveReferencedSearchResult(
            originalArgs?.url,
            results
        );


    if(
        !searchResult
    ){

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
     * CANDIDATES
     * =====================================================
     */


    const candidates =

        scopeCandidatesToSearchQuery(
            searchResult?.arguments?.query,
            extractSearchCandidates(searchResult)
        );


    if(
        candidates.length === 0
    ){

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

                selectionContextForSearch(selectionContext, searchResult),

                candidates

            );


    }catch(error){


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


    if(
        selection?.noSuitableSource
        ===
        true
    ){

        return buildFailure({

            failureType:
                "no-suitable-source",

            shouldRetry:
                true,

            text:

                selection.reason
                ||
                "Не найден подходящий источник"

        });

    }


    /*
     * =====================================================
     * SELECTOR FAILURE
     * =====================================================
     */


    if(
        selection?.success !== true
        ||
        !selection?.result?.url
    ){

        return buildFailure({

            failureType:
                "source-selector-error",

            text:

                selection?.reason
                ||
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
