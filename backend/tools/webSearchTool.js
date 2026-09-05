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
 * Planner решает, КОГДА нужен поиск.
 * Этот инструмент только выполняет его.
 */


const tinyFishApiKey =
    process.env.TINYFISH_API_KEY || "";


const tinyFishSearchUrl =
    "https://api.search.tinyfish.ai";


/*
 * =========================================================
 * NORMALIZE RESULT
 * =========================================================
 */


function normalizeSearchResult(
    item
) {

    return {

        title:
            typeof item?.title === "string"
                ? item.title
                : "",

        url:
            typeof item?.url === "string"
                ? item.url
                : "",

        snippet:
            typeof item?.snippet === "string"
                ? item.snippet
                : (
                    typeof item?.description === "string"
                        ? item.description
                        : ""
                )

    };

}


/*
 * =========================================================
 * SEARCH
 * =========================================================
 */


async function executeWebSearch(
    args
) {

    const query =
        typeof args?.query === "string"
            ? args.query.trim()
            : "";


    if (!query) {

        return {
            success: false,

            needsClarification: true,

            text:
                "Не указан поисковый запрос."
        };

    }


    if (!tinyFishApiKey) {

        return {
            success: false,

            text:
                "Интернет-поиск Jessica не настроен."
        };

    }


    try {

        const url =
            new URL(
                tinyFishSearchUrl
            );


        url.searchParams.set(
            "query",
            query
        );


        const response =
            await fetch(
                url,
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


        if (!response.ok) {

            console.error(
                "TinyFish Search error:",
                response.status,
                rawText
            );


            return {
                success: false,

                text:
                    `Интернет-поиск завершился ошибкой HTTP ${response.status}.`
            };

        }


        let data;


        try {

            data =
                JSON.parse(
                    rawText
                );

        } catch {

            console.error(
                "TinyFish invalid JSON:",
                rawText
            );


            return {
                success: false,

                text:
                    "Интернет-поиск вернул некорректные данные."
            };

        }


        const rawResults =
            Array.isArray(
                data.results
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
                        item.title ||
                        item.url ||
                        item.snippet
                )
                .slice(
                    0,
                    8
                );


        if (
            results.length === 0
        ) {

            return {
                success: false,

                text:
                    `По запросу «${query}» не найдено подходящих результатов.`,

                data: {
                    query,
                    results: []
                }
            };

        }


        return {
            success: true,

            text:
                `Найдено результатов: ${results.length}.`,

            data: {
                query,
                results
            }
        };


    } catch (error) {

        console.error(
            "Web Search Tool error:",
            error
        );


        return {
            success: false,

            text:
                "Jessica не смогла выполнить интернет-поиск."
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
            "Ищет актуальную информацию в интернете. " +
            "Используй для новостей, погоды, цен, расписаний, " +
            "текущих событий, актуального состояния объектов, " +
            "поиска сайтов и другой информации, которая может изменяться."
        ),

    arguments: {

        query:
            (
                "Самостоятельный поисковый запрос. " +
                "Он должен содержать всю информацию, необходимую для поиска."
            )

    },

    execute:
        executeWebSearch

});
