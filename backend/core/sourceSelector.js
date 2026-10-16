import {
    isSourceSelectorAvailable,
    requestSourceSelection
} from "../ai/sourceSelectorClient.js";


import {
    buildSourceSelectionInstructions
} from "./source/sourceSelectionPrompt.js";


/*
 * =========================================================
 * JESSICA SOURCE SELECTOR v2
 * =========================================================
 *
 * Search Results
 *      ↓
 * Normalize
 *      ↓
 * AI Source Selector Client
 *      ↓
 * Parse
 *      ↓
 * Selected Source
 *
 *
 * AI transport:
 *
 * backend/ai/sourceSelectorClient.js
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
            (
                item,
                index
            ) => ({

                index,

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
 * NO SUITABLE SOURCE
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
 * PARSE DECISION
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
     * =====================================================
     * REJECT
     * =====================================================
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
     * =====================================================
     * SELECT
     * =====================================================
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
        !Number.isInteger(index)
        ||
        index < 0
        ||
        index >= results.length
    ){

        return noSuitableSource(

            "Source Selector указал некорректный источник"

        );

    }


    return {

        success:
            true,

        noSuitableSource:
            false,

        index,

        result:
            results[index],

        reason:

            typeof parsed?.reason === "string"

                ? parsed.reason.trim()

                : ""

    };

}


/*
 * =========================================================
 * SELECT SOURCE
 * =========================================================
 */


export async function selectSource(
    task,
    searchResults
) {

    const results =

        normalizeResults(
            searchResults
        );


    /*
     * =====================================================
     * NO RESULTS
     * =====================================================
     */


    if(
        results.length === 0
    ){

        return noSuitableSource(
            "Поиск не вернул источники"
        );

    }


    /*
     * =====================================================
     * AI AVAILABILITY
     * =====================================================
     */


    if(
        !isSourceSelectorAvailable()
    ){

        return noSuitableSource(
            "Source Selector недоступен"
        );

    }


    /*
     * =====================================================
     * REQUEST
     * =====================================================
     */


    try {


        const response =

            await requestSourceSelection({

                instructions:

                    buildSourceSelectionInstructions(),

                task:

                    String(
                        task || ""
                    )
                        .trim(),

                results

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

                results

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

                    String(error)

            }

        );


        return noSuitableSource(

            error?.message

            ||

            "Не удалось выбрать надёжный источник"

        );

    }

}
