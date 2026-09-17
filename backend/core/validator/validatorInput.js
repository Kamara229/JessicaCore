/*
 * =========================================================
 * JESSICA VALIDATOR INPUT
 * =========================================================
 *
 * Подготавливает компактный контекст
 * для AI Result Validator.
 *
 *
 * Задача:
 *
 * большой TaskRunner result
 *        ↓
 * compact validator input
 *        ↓
 * AI Validator
 *
 *
 * Нужен для защиты от слишком больших prompt:
 *
 * 413 Request too large
 *
 *
 * Этот модуль НЕ:
 *
 * - вызывает AI;
 * - валидирует ответ;
 * - меняет результат выполнения;
 * - выбирает источники;
 * - строит новый план.
 *
 * =========================================================
 */


/*
 * =========================================================
 * LIMITS
 * =========================================================
 */


const MAX_SEARCH_RESULTS =
    5;


const MAX_SNIPPET_LENGTH =
    500;


const MAX_TEXT_LENGTH =
    2500;


const MAX_GENERIC_STRING_LENGTH =
    800;


/*
 * =========================================================
 * SAFE STRING
 * =========================================================
 */


function safeString(
    value,
    maxLength = MAX_GENERIC_STRING_LENGTH
) {

    const text =
        typeof value === "string"
            ? value.trim()
            : "";


    if (
        !text
    ) {

        return "";

    }


    if (
        text.length <= maxLength
    ) {

        return text;

    }


    return (
        text.slice(
            0,
            maxLength
        ) +
        "…"
    );

}


/*
 * =========================================================
 * SEARCH RESULT
 * =========================================================
 */


function compactSearchResult(
    item
) {

    if (
        !item ||
        typeof item !== "object"
    ) {

        return null;

    }


    const compact = {

        title:
            safeString(
                item.title,
                300
            ),

        url:
            safeString(
                item.url,
                500
            ),

        snippet:
            safeString(

                item.snippet ||
                item.description ||
                item.text,

                MAX_SNIPPET_LENGTH

            )

    };


    return compact;

}


/*
 * =========================================================
 * SEARCH RESULTS
 * =========================================================
 */


function compactSearchResults(
    results
) {

    if (
        !Array.isArray(
            results
        )
    ) {

        return [];

    }


    return results
        .slice(
            0,
            MAX_SEARCH_RESULTS
        )
        .map(
            compactSearchResult
        )
        .filter(Boolean);

}


/*
 * =========================================================
 * WEB SEARCH DATA
 * =========================================================
 */


function compactWebSearchData(
    data
) {

    if (
        !data ||
        typeof data !== "object"
    ) {

        return null;

    }


    return {

        query:
            safeString(
                data.query,
                500
            ),

        results:
            compactSearchResults(
                data.results
            )

    };

}


/*
 * =========================================================
 * WEB FETCH DATA
 * =========================================================
 */


function compactWebFetchData(
    data
) {

    if (
        !data ||
        typeof data !== "object"
    ) {

        return null;

    }


    return {

        url:
            safeString(
                data.url,
                500
            ),

        title:
            safeString(
                data.title,
                300
            ),

        text:
            safeString(

                data.text ||
                data.content ||
                data.body,

                MAX_TEXT_LENGTH

            )

    };

}


/*
 * =========================================================
 * GENERIC VALUE
 * =========================================================
 */


function compactGenericValue(
    value,
    depth = 0
) {

    /*
     * Не уходим глубоко в неизвестные payload.
     */

    if (
        depth > 3
    ) {

        return "[truncated]";

    }


    if (
        value === null ||
        value === undefined
    ) {

        return value;

    }


    if (
        typeof value === "string"
    ) {

        return safeString(
            value
        );

    }


    if (
        typeof value === "number" ||
        typeof value === "boolean"
    ) {

        return value;

    }


    if (
        Array.isArray(
            value
        )
    ) {

        return value
            .slice(
                0,
                10
            )
            .map(
                item =>
                    compactGenericValue(
                        item,
                        depth + 1
                    )
            );

    }


    if (
        typeof value === "object"
    ) {

        const result =
            {};


        const entries =
            Object.entries(
                value
            )
                .slice(
                    0,
                    20
                );


        for (
            const [
                key,
                nestedValue
            ]
            of entries
        ) {

            result[key] =
                compactGenericValue(

                    nestedValue,

                    depth + 1

                );

        }


        return result;

    }


    return String(
        value
    );

}


/*
 * =========================================================
 * TOOL RESULT
 * =========================================================
 */


function compactToolResult(
    item
) {

    if (
        !item ||
        typeof item !== "object"
    ) {

        return null;

    }


    const tool =
        safeString(
            item.tool,
            100
        );


    /*
     * =====================================================
     * WEB SEARCH
     * =====================================================
     */


    if (
        tool === "web_search"
    ) {

        return {

            stepId:
                safeString(
                    item.stepId ||
                    item.id,
                    100
                ),

            tool,

            success:
                item.success !== false,

            data:
                compactWebSearchData(
                    item.data
                )

        };

    }


    /*
     * =====================================================
     * WEB FETCH
     * =====================================================
     */


    if (
        tool === "web_fetch"
    ) {

        return {

            stepId:
                safeString(
                    item.stepId ||
                    item.id,
                    100
                ),

            tool,

            success:
                item.success !== false,

            data:
                compactWebFetchData(
                    item.data
                )

        };

    }


    /*
     * =====================================================
     * GENERIC TOOL
     * =====================================================
     */


    return {

        stepId:
            safeString(
                item.stepId ||
                item.id,
                100
            ),

        tool,

        success:
            item.success !== false,

        reason:
            safeString(
                item.reason ||
                item.text
            ),

        data:
            compactGenericValue(
                item.data
            )

    };

}


/*
 * =========================================================
 * COMPACT TASK RUN RESULT
 * =========================================================
 */


export function buildCompactValidatorInput(
    taskRunResult
) {

    if (
        !taskRunResult ||
        typeof taskRunResult !== "object"
    ) {

        return taskRunResult;

    }


    const results =
        Array.isArray(
            taskRunResult.results
        )
            ? taskRunResult.results
            : [];


    return {

        success:
            taskRunResult.success !== false,


        needsClarification:
            taskRunResult.needsClarification === true,


        shouldRetry:
            taskRunResult.shouldRetry === true,


        reason:
            safeString(

                taskRunResult.reason ||
                taskRunResult.text

            ),


        stage:
            safeString(
                taskRunResult.stage,
                100
            ),


        failureType:
            safeString(
                taskRunResult.failureType,
                100
            ),


        results:
            results
                .map(
                    compactToolResult
                )
                .filter(Boolean)

    };

              }
