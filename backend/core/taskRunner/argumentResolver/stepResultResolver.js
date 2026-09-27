/*
 * =========================================================
 * JESSICA TASK RUNNER
 * STEP RESULT RESOLVER v2
 * =========================================================
 *
 * Работа с результатами предыдущих Execution Steps.
 *
 *
 * Отвечает:
 *
 * - поиск результата шага;
 * - поиск результата web_search;
 * - извлечение URL;
 * - подготовка кандидатов источников.
 *
 *
 * НЕ:
 *
 * - выполняет Tools;
 * - разрешает arguments;
 * - делает Retry;
 * - делает Replan.
 *
 * =========================================================
 */







/*
 * =========================================================
 * URL VALIDATION
 * =========================================================
 */


function isValidUrl(

    value

) {


    if (

        typeof value !== "string"

    ) {

        return false;

    }





    return /^https?:\/\/\S+$/i.test(

        value.trim()

    );

}









/*
 * =========================================================
 * FIND STEP RESULT
 * =========================================================
 */


export function findStepResult(

    stepId,

    results

) {


    if (

        typeof stepId !== "string"

        ||

        !stepId.trim()

        ||

        !Array.isArray(results)

    ) {

        return null;

    }







    return (

        results.find(

            item =>

                item?.id === stepId.trim()

        )

        ||

        null

    );

}









/*
 * =========================================================
 * CHECK TOOL NAME
 * =========================================================
 */


function isWebSearchResult(

    item

) {


    return (

        item?.tool === "web_search"

        ||

        item?.metadata?.tool === "web_search"

        ||

        item?.name === "web_search"

    );

}









/*
 * =========================================================
 * FIND LAST SEARCH RESULT
 * =========================================================
 */


export function findPreviousSearchResult(

    results

) {


    if (

        !Array.isArray(results)

    ) {

        return null;

    }







    const searches =

        results.filter(

            item =>

                isWebSearchResult(item)

                &&

                item?.success === true


        );







    return (

        searches.length > 0

            ?

            searches[searches.length - 1]

            :

            null

    );

}









/*
 * =========================================================
 * EXTRACT URL
 * =========================================================
 */


export function extractUrl(

    item

) {


    if (

        !item

        ||

        typeof item !== "object"

    ) {

        return null;

    }







    const candidates = [


        item.url,


        item.link,


        item.href,


        item.sourceUrl,


        item.source_url


    ];








    for (

        const value

        of candidates

    ) {


        if (

            isValidUrl(value)

        ) {

            return value.trim();

        }

    }







    return null;

}









/*
 * =========================================================
 * EXTRACT SEARCH RESULTS ARRAY
 * =========================================================
 */


function getSearchResults(

    searchResult

) {


    return (

        searchResult?.data?.results

        ||

        searchResult?.results

        ||

        searchResult?.data

        ||

        []

    );

}









/*
 * =========================================================
 * EXTRACT SEARCH CANDIDATES
 * =========================================================
 */


export function extractSearchCandidates(

    searchResult

) {


    const results =

        getSearchResults(

            searchResult

        );







    if (

        !Array.isArray(results)

    ) {

        return [];

    }







    return results

        .map(

            item => {


                const url =

                    extractUrl(item);



                return {


                    ...item,


                    url


                };


            }

        )


        .filter(

            item =>

                Boolean(

                    item.url

                )

        );

}









/*
 * =========================================================
 * HAS SOURCES
 * =========================================================
 */


export function hasSearchSources(

    searchResult

) {


    return (

        extractSearchCandidates(

            searchResult

        )
        .length > 0

    );

}









/*
 * =========================================================
 * GET SOURCE URLS
 * =========================================================
 */


export function extractSourceUrls(

    searchResult

) {


    return (

        extractSearchCandidates(

            searchResult

        )

        .map(

            item =>

                item.url

        )

    );

}
