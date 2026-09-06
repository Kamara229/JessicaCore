import {
    registerTool
} from "./toolRegistry.js";


/*
 * =========================================================
 * JESSICA WEB SEARCH TOOL
 * =========================================================
 *
 * Универсальный интернет-поиск Jessica.
 *
 * Использует TinyFish Search.
 *
 * ВАЖНО:
 *
 * web_search НЕ формирует конечный ответ пользователю.
 *
 * Он только получает поисковые данные,
 * которые затем анализирует Answer Composer.
 *
 * Поэтому успешный результат хранится
 * в data.results, а text остаётся пустым.
 */


/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


const tinyFishApiKey =
    process.env.TINYFISH_API_KEY || "";


const tinyFishSearchUrl =
    "https://api.search.tinyfish.ai";


const MAX_RESULTS =
    8;


/*
 * =========================================================
 * NORMALIZE SEARCH RESULT
 * =========================================================
 */


function normalizeSearchResult(
    item
) {

    if (
        !item ||
        typeof item !== "object"
    ) {

        return null;

    }


    const title =
        typeof item.title === "string"
            ? item.title.trim()
            : "";


    const url =
        typeof item.url === "string"
            ? item.url.trim()
            : "";


    const snippet =
        typeof item.snippet === "string"
            ? item.snippet.trim()
            : (
                typeof item.description === "string"
                    ? item.description.trim()
                    : ""
            );


    if (
        !title &&
        !url &&
        !snippet
    ) {

        return null;

    }


    return {
        title,
        url,
        snippet
    };

}


/*
 * =========================================================
 * EXECUTE WEB SEARCH
 * =========================================================
 */


async function executeWebSearch(
    args
) {

    const query =
        typeof args?.query === "string"
            ? args.query.trim()
            : "";


    /*
     * -----------------------------------------------------
     * INPUT VALIDATION
     * -----------------------------------------------------
     */


    if (!query) {

        return {
            success: false,

            needsClarification:
                true,

            text:
                "Не указан поисковый запрос.",

            data:
                null
        };

    }


    if (!tinyFishApiKey) {

        return {
            success: false,

            text:
                "Интернет-поиск Jessica не настроен.",

            data:
                null
        };

    }


    /*
     * -----------------------------------------------------
     * REQUEST
     * -----------------------------------------------------
     */


    try {

        const requestUrl =
            new URL(
                tinyFishSearchUrl
            );


        requestUrl.searchParams.set(
            "query",
            query
        );


        const response =
            await fetch(
                requestUrl,
                {
                    method:
                        "GET",

                    headers: {
                        "X-API-Key":
                            tinyFishApiKey
                    },

                    signal:
                        AbortSignal.timeout(
                            30000
                        )
                }
            );


        const rawText =
            await response.text();


        /*
         * -------------------------------------------------
         * HTTP ERROR
         * -------------------------------------------------
         */


        if (!response.ok) {

            console.error(
                "TinyFish Search error:",
                response.status,
                rawText
            );


            return {
                success: false,

                text:
                    `Интернет-поиск завершился ошибкой HTTP ${response.status}.`,

                data:
                    null
            };

        }


        /*
         * -------------------------------------------------
         * PARSE JSON
         * -------------------------------------------------
         */


        let data;


        try {

            data =
                JSON.parse(
                    rawText
                );

        } catch (error) {

            console.error(
                "TinyFish Search invalid JSON:",
                rawText
            );


            return {
                success: false,

                text:
                    "Интернет-поиск вернул некорректные данные.",

                data:
                    null
            };

        }


        /*
         * -------------------------------------------------
         * NORMALIZE RESULTS
         * -------------------------------------------------
         */


        const rawResults =
            Array.isArray(
                data?.results
            )
                ? data.results
                : [];


        const results =
            rawResults
                .map(
                    normalizeSearchResult
                )
                .filter(
                    item =>
                        item !== null
                )
                .slice(
                    0,
                    MAX_RESULTS
                );


        /*
         * -------------------------------------------------
         * NOTHING FOUND
         * -------------------------------------------------
         */


        if (
            results.length === 0
        ) {

            return {
                success: false,

                text:
                    `По запросу «${query}» не найдено подходящих результатов.`,

                data: {
                    query,
                    resultCount:
                        0,
                    results:
                        []
                }
            };

        }


        /*
         * -------------------------------------------------
         * SUCCESS
         * -------------------------------------------------
         *
         * ВАЖНО:
         *
         * text специально пустой.
         *
         * Это НЕ готовый пользовательский ответ.
         *
         * Answer Composer должен изучить data.results
         * и самостоятельно сформировать нормальный ответ.
         */


        return {
            success: true,

            text:
                "",

            data: {
                query,

                resultCount:
                    results.length,

                results
            }
        };


    } catch (error) {

        console.error(
            "Web Search Tool exception:",
            error
        );


        /*
         * AbortSignal.timeout()
         */
        if (
            error?.name ===
            "TimeoutError"
        ) {

            return {
                success: false,

                text:
                    "Интернет-поиск превысил допустимое время ожидания.",

                data:
                    null
            };

        }


        return {
            success: false,

            text:
                "Jessica не смогла выполнить интернет-поиск.",

            data:
                null
        };

    }

}


/*
 * =========================================================
 * REGISTER TOOL
 * =========================================================
 */


registerTool({

    name:
        "web_search",

    description:
        (
            "Выполняет поиск актуальной информации в интернете " +
            "и возвращает результаты поиска для дальнейшего анализа Jessica. " +

            "Используй, когда для решения задачи требуются свежие, " +
            "изменяющиеся или внешние данные, которых нельзя надёжно получить " +
            "только из знаний AI. " +

            "Инструмент не формирует конечный ответ пользователю."
        ),

    arguments: {

        query:
            (
                "Полноценный самостоятельный поисковый запрос, " +
                "содержащий необходимые названия, условия, даты " +
                "и другие важные сведения из задачи пользователя."
            )

    },

    execute:
        executeWebSearch

});
