/*
 * =========================================================
 * JESSICA TASK RUNNER
 * STEP RESULT RESOLVER v1
 * =========================================================
 *
 * Работа с результатами предыдущих Execution Steps.
 *
 *
 * Отвечает:
 *
 * - поиск результата шага;
 * - поиск web_search результата;
 * - извлечение URL;
 * - подготовка источников.
 *
 *
 * НЕ:
 *
 * - выполняет Tools;
 * - делает Retry;
 * - делает Replan;
 * - разрешает аргументы;
 *
 * =========================================================
 */







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

    ) {

        return null;

    }





    if (

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


                item?.tool === "web_search"

                &&

                item?.success === true


        );







    if (

        searches.length === 0

    ) {

        return null;

    }







    return searches[

        searches.length - 1

    ];

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







    return (

        item.url

        ||

        item.link

        ||

        item.href

        ||

        null

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


        searchResult?.data?.results

        ||

        searchResult?.results

        ||

        [];







    if (

        !Array.isArray(results)

    ) {

        return [];

    }







    return results

        .map(

            item =>


            ({

                ...item,


                url:

                    extractUrl(item)

            })


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
 * HAS SEARCH SOURCES
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
