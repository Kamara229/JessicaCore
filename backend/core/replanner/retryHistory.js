/*
 * =========================================================
 * JESSICA REPLANNER
 * RETRY HISTORY v2
 * =========================================================
 *
 * Собирает компактную историю предыдущих
 * маршрутов Execution.
 *
 *
 * Теперь сохраняет:
 *
 * - search queries;
 * - реально использованные tools;
 * - failed tools;
 * - реально открытые source URLs;
 * - накопительную историю source URLs;
 * - evidence mode предыдущего плана.
 *
 *
 * Главная задача:
 *
 * не позволять Replanner ходить
 * по одним и тем же источникам по кругу.
 *
 * =========================================================
 */


/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


const MAX_TRIED_SEARCH_QUERIES =
    8;


const MAX_TRIED_SOURCE_URLS =
    12;


/*
 * =========================================================
 * SAFE STRING
 * =========================================================
 */


function safeString(
    value
) {

    return typeof value === "string"

        ? value.trim()

        : "";

}


/*
 * =========================================================
 * UNIQUE STRINGS
 * =========================================================
 */


function uniqueStrings(
    values
) {

    if(
        !Array.isArray(
            values
        )
    ){

        return [];

    }


    return [

        ...new Set(

            values

                .map(
                    item =>
                        safeString(
                            item
                        )
                )

                .filter(Boolean)

        )

    ];

}


/*
 * =========================================================
 * EXTRACT SEARCH QUERIES
 * =========================================================
 */


export function extractSearchQueries(
    plan
) {

    if(
        !Array.isArray(
            plan?.steps
        )
    ){

        return [];

    }


    const queries =

        plan.steps

            .filter(
                step =>
                    step?.tool === "web_search"
            )

            .map(
                step =>
                    safeString(
                        step
                            ?.arguments
                            ?.query
                    )
            );


    return uniqueStrings(
        queries
    );

}


/*
 * =========================================================
 * EXTRACT USED TOOLS
 * =========================================================
 */


export function extractUsedTools(
    runResult
) {

    if(
        !Array.isArray(
            runResult?.results
        )
    ){

        return [];

    }


    return uniqueStrings(

        runResult.results.map(
            item =>
                item?.tool
        )

    );

}


/*
 * =========================================================
 * EXTRACT FAILED TOOLS
 * =========================================================
 */


export function extractFailedTools(
    runResult
) {

    if(
        !Array.isArray(
            runResult?.results
        )
    ){

        return [];

    }


    return uniqueStrings(

        runResult.results

            .filter(
                item =>
                    item?.success === false
            )

            .map(
                item =>
                    item?.tool
            )

    );

}


/*
 * =========================================================
 * EXTRACT FETCHED URL
 * =========================================================
 */


function extractResultUrl(
    result
) {

    if(
        !result
        ||
        typeof result !== "object"
    ){

        return "";

    }


    const candidates = [

        result?.data?.url,

        result?.url,

        result?.arguments?.url,

        result?.input?.url,

        result?.resolvedArguments?.url

    ];


    for(
        const candidate
        of candidates
    ){

        const url =

            safeString(
                candidate
            );


        if(
            url
        ){

            return url;

        }

    }


    return "";

}


/*
 * =========================================================
 * EXTRACT FETCHED URLS
 * =========================================================
 *
 * Берём фактически выполненные web_fetch.
 *
 * Это важнее URL из Plan,
 * потому что Plan может содержать $from.
 *
 * =========================================================
 */


export function extractFetchedUrls(
    runResult
) {

    if(
        !Array.isArray(
            runResult?.results
        )
    ){

        return [];

    }


    return uniqueStrings(

        runResult.results

            .filter(
                item =>
                    item?.tool === "web_fetch"
            )

            .map(
                extractResultUrl
            )

    );

}


/*
 * =========================================================
 * EXTRACT PLAN FETCH URLS
 * =========================================================
 *
 * Fallback для прямых URL,
 * указанных самим Planner.
 *
 * $from здесь автоматически
 * не превращается в строку.
 *
 * =========================================================
 */


export function extractPlanFetchUrls(
    plan
) {

    if(
        !Array.isArray(
            plan?.steps
        )
    ){

        return [];

    }


    return uniqueStrings(

        plan.steps

            .filter(
                step =>
                    step?.tool === "web_fetch"
            )

            .map(
                step => {

                    const value =
                        step?.arguments?.url;


                    return typeof value === "string"

                        ? value

                        : "";

                }
            )

    );

}


/*
 * =========================================================
 * PREVIOUS ATTEMPT
 * =========================================================
 */


export function buildPreviousAttempt(
    previousPlan,
    previousRunResult
) {

    const fetchedUrls =

        uniqueStrings([

            ...extractFetchedUrls(
                previousRunResult
            ),

            ...extractPlanFetchUrls(
                previousPlan
            )

        ]);


    return {

        intent:

            safeString(
                previousPlan?.intent
            ),


        evidenceMode:

            safeString(
                previousPlan
                    ?.evidence
                    ?.mode
            ),


        stepsCount:

            Array.isArray(
                previousPlan?.steps
            )

                ? previousPlan.steps.length

                : 0,


        searchQueries:

            extractSearchQueries(
                previousPlan
            ),


        sourceUrls:

            fetchedUrls,


        usedTools:

            extractUsedTools(
                previousRunResult
            ),


        failedTools:

            extractFailedTools(
                previousRunResult
            )

    };

}


/*
 * =========================================================
 * TRIED SEARCH QUERIES
 * =========================================================
 */


export function buildTriedSearchQueries(
    planningContext,
    previousPlan
) {

    const existing =

        Array.isArray(
            planningContext
                ?.metadata
                ?.triedSearchQueries
        )

            ? planningContext
                .metadata
                .triedSearchQueries

            : [];


    const current =

        extractSearchQueries(
            previousPlan
        );


    return uniqueStrings([

        ...existing,

        ...current

    ])
        .slice(
            -MAX_TRIED_SEARCH_QUERIES
        );

}


/*
 * =========================================================
 * TRIED SOURCE URLS
 * =========================================================
 */


export function buildTriedSourceUrls(

    planningContext,

    previousPlan,

    previousRunResult

) {

    const existing =

        Array.isArray(
            planningContext
                ?.metadata
                ?.triedSourceUrls
        )

            ? planningContext
                .metadata
                .triedSourceUrls

            : [];


    const current =

        uniqueStrings([

            ...extractFetchedUrls(
                previousRunResult
            ),

            ...extractPlanFetchUrls(
                previousPlan
            )

        ]);


    return uniqueStrings([

        ...existing,

        ...current

    ])
        .slice(
            -MAX_TRIED_SOURCE_URLS
        );

}


/*
 * =========================================================
 * BUILD RETRY HISTORY
 * =========================================================
 */


export function buildRetryHistory({

    planningContext = {},

    previousPlan = null,

    previousRunResult = null

} = {}) {


    return {

        previousAttempt:

            buildPreviousAttempt(

                previousPlan,

                previousRunResult

            ),


        triedSearchQueries:

            buildTriedSearchQueries(

                planningContext,

                previousPlan

            ),


        triedSourceUrls:

            buildTriedSourceUrls(

                planningContext,

                previousPlan,

                previousRunResult

            )

    };

}
