import {
    isSourceSelectorAvailable,
    requestSourceSelection
} from "../ai/sourceSelectorClient.js";


import {
    buildSourceSelectionInstructions
} from "./source/sourceSelectionPrompt.js";


/*
 * =========================================================
 * JESSICA SOURCE SELECTOR v3
 * =========================================================
 *
 * Search Results
 *      ↓
 * Normalize
 *      ↓
 * MACHINE EXCLUSION
 *      ↓
 * AI chooses candidate
 *      ↓
 * Parse
 *
 *
 * КРИТИЧЕСКО:
 *
 * rejected URL удаляется ДО AI.
 *
 * AI reason не является evidence.
 *
 * =========================================================
 */


/*
 * =========================================================
 * CLEAN JSON
 * =========================================================
 */


function cleanJsonText(
    text
) {

    let value =

        String(
            text || ""
        )
            .replace(
                /```json/gi,
                ""
            )
            .replace(
                /```/g,
                ""
            )
            .trim();


    const firstBrace =
        value.indexOf("{");


    const lastBrace =
        value.lastIndexOf("}");


    if(
        firstBrace !== -1
        &&
        lastBrace > firstBrace
    ){

        value =

            value.slice(

                firstBrace,

                lastBrace + 1

            );

    }


    return value;

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
 * NORMALIZE SELECTION CONTEXT
 * =========================================================
 *
 * Backward compatible:
 *
 * string
 *      → task only
 *
 * object
 *      → task + exclusions
 *
 * =========================================================
 */


function normalizeSelectionContext(
    context
) {

    if(
        typeof context === "string"
    ){

        return {

            task:
                context.trim(),

            excludedSourceUrls:
                []

        };

    }


    if(
        !context
        ||
        typeof context !== "object"
        ||
        Array.isArray(
            context
        )
    ){

        return {

            task:
                "",

            excludedSourceUrls:
                []

        };

    }


    return {

        task:

            String(
                context.task || ""
            )
                .trim(),


        excludedSourceUrls:

            Array.isArray(
                context.excludedSourceUrls
            )

                ? [
                    ...context.excludedSourceUrls
                ]

                : []

    };

}


/*
 * =========================================================
 * NORMALIZE RESULTS
 * =========================================================
 */


function normalizeResults(
    results
) {

    if(
        !Array.isArray(
            results
        )
    ){

        return [];

    }


    return results

        .map(
            item => ({

                title:

                    String(
                        item?.title || ""
                    )
                        .trim(),

                url:

                    String(
                        item?.url || ""
                    )
                        .trim(),

                snippet:

                    String(
                        item?.snippet || ""
                    )
                        .trim()

            })
        )

        .filter(
            item =>
                item.url
        );

}


/*
 * =========================================================
 * MACHINE FILTER
 * =========================================================
 */


function filterExcludedResults(
    results,
    excludedUrls
) {

    const excluded =

        new Set(

            excludedUrls

                .map(
                    normalizeUrl
                )

                .filter(Boolean)

        );


    const filtered =
        [];


    let excludedCount =
        0;


    for(
        const item
        of results
    ){

        if(
            excluded.has(
                normalizeUrl(
                    item.url
                )
            )
        ){

            excludedCount++;

            continue;

        }


        filtered.push(
            item
        );

    }


    /*
     * AI indexes filtered array.
     */


    const indexed =

        filtered.map(
            (
                item,
                index
            ) => ({

                index,

                ...item

            })
        );


    return {

        results:
            indexed,

        excludedCount

    };

}


/*
 * =========================================================
 * NO SOURCE
 * =========================================================
 */


function noSuitableSource(
    reason
) {

    return {

        success:
            false,

        noSuitableSource:
            true,

        index:
            null,

        result:
            null,

        reason:

            String(

                reason
                ||
                "Подходящий источник не найден"

            )
                .trim()

    };

}


/*
 * =========================================================
 * PARSE
 * =========================================================
 */


function parseSelection(
    raw,
    results
) {

    if(
        !raw
    ){

        return noSuitableSource(
            "Source Selector вернул пустой ответ"
        );

    }


    let parsed;


    try {


        parsed =

            JSON.parse(

                cleanJsonText(
                    raw
                )

            );


    }catch(error){


        console.error(

            "Source Selector invalid JSON:",

            raw

        );


        return noSuitableSource(

            "Source Selector не смог надёжно оценить поисковую выдачу"

        );

    }


    const decision =

        String(
            parsed?.decision || ""
        )
            .trim()
            .toLowerCase();


    /*
     * REJECT.
     */


    if(
        decision === "reject"
    ){

        return noSuitableSource(

            parsed?.reason
            ||
            "Ни один найденный источник не удовлетворяет задаче"

        );

    }


    /*
     * SELECT.
     */


    if(
        decision !== "select"
    ){

        return noSuitableSource(

            "Source Selector вернул неизвестное решение"

        );

    }


    const index =

        Number(
            parsed?.index
        );


    if(
        !Number.isInteger(
            index
        )
        ||
        index < 0
        ||
        index >= results.length
    ){

        return noSuitableSource(

            "Source Selector указал некорректный источник"

        );

    }


    /*
     * Не используем AI reason как evidence.
     *
     * Даже если модель написала:
     *
     * "страница содержит дату"
     *
     * это не переносится дальше.
     */


    return {

        success:
            true,

        noSuitableSource:
            false,

        index,

        result:
            results[index],

        reason:

            (
                "Выбран кандидат по title, URL и snippet поисковой выдачи; "
                +
                "содержимое страницы ещё не проверено."
            )

    };

}


/*
 * =========================================================
 * SELECT SOURCE
 * =========================================================
 */


export async function selectSource(
    selectionContext,
    searchResults
) {

    const context =

        normalizeSelectionContext(
            selectionContext
        );


    const normalizedResults =

        normalizeResults(
            searchResults
        );


    /*
     * =====================================================
     * NO SEARCH RESULTS
     * =====================================================
     */


    if(
        normalizedResults.length === 0
    ){

        return noSuitableSource(
            "Поиск не вернул источники"
        );

    }


    /*
     * =====================================================
     * MACHINE EXCLUSION
     * =====================================================
     */


    const filtered =

        filterExcludedResults(

            normalizedResults,

            context.excludedSourceUrls

        );


    if(
        filtered.excludedCount > 0
    ){

        console.log(

            "Jessica Source Selector exclusions applied:",

            JSON.stringify({

                excluded:
                    filtered.excludedCount,

                remaining:
                    filtered.results.length

            })

        );

    }


    if(
        filtered.results.length === 0
    ){

        return noSuitableSource(

            "Все найденные источники уже были проверены и отклонены"

        );

    }


    /*
     * =====================================================
     * AI
     * =====================================================
     */


    if(
        !isSourceSelectorAvailable()
    ){

        return noSuitableSource(
            "Source Selector недоступен"
        );

    }


    try {


        const response =

            await requestSourceSelection({

                instructions:

                    buildSourceSelectionInstructions(),

                task:

                    context.task,

                results:

                    filtered.results

            });


        const raw =

            response
                ?.choices
                ?.[0]
                ?.message
                ?.content;


        const selection =

            parseSelection(

                raw,

                filtered.results

            );


        /*
         * =================================================
         * LOG
         * =================================================
         */


        if(
            selection.success
        ){

            console.log(

                "Jessica Source Selector:",

                JSON.stringify({

                    decision:
                        "select",

                    index:
                        selection.index,

                    title:
                        selection.result?.title || "",

                    url:
                        selection.result?.url || "",

                    reason:
                        selection.reason || ""

                })

            );

        }else{


            console.warn(

                "Jessica Source Selector:",

                JSON.stringify({

                    decision:
                        "reject",

                    reason:
                        selection.reason || ""

                })

            );

        }


        return selection;


    }catch(error){


        console.error(

            "Source Selector final error:",

            {

                status:
                    error?.status || 0,

                code:
                    error?.code || null,

                message:

                    error?.message
                    ||
                    String(
                        error
                    )

            }

        );


        return noSuitableSource(

            error?.message
            ||
            "Не удалось выбрать надёжный источник"

        );

    }

}
