import {
    registerTool
} from "./toolRegistry.js";


/*
 * =========================================================
 * JESSICA WEB FETCH TOOL
 * =========================================================
 *
 * Загружает содержимое конкретной веб-страницы.
 *
 * Использует TinyFish Fetch.
 *
 * Planner решает, когда нужно открыть страницу.
 * Этот инструмент только получает её содержимое.
 */


const tinyFishApiKey =
    process.env.TINYFISH_API_KEY || "";


const tinyFishFetchUrl =
    "https://api.fetch.tinyfish.ai";


/*
 * =========================================================
 * FETCH PAGE
 * =========================================================
 */


async function executeWebFetch(
    args
) {

    const pageUrl =
        typeof args?.url === "string"
            ? args.url.trim()
            : "";


    if (!pageUrl) {

        return {
            success: false,

            needsClarification: true,

            text:
                "Не указан адрес страницы."
        };

    }


    /*
     * Проверяем URL до отправки
     * запроса во внешний сервис.
     */
    let normalizedUrl;


    try {

        normalizedUrl =
            new URL(
                pageUrl
            );


        if (
            normalizedUrl.protocol !== "http:" &&
            normalizedUrl.protocol !== "https:"
        ) {

            return {
                success: false,

                text:
                    "Jessica может открывать только HTTP и HTTPS страницы."
            };

        }

    } catch {

        return {
            success: false,

            text:
                "Указан некорректный адрес страницы."
        };

    }


    if (!tinyFishApiKey) {

        return {
            success: false,

            text:
                "Загрузка веб-страниц Jessica не настроена."
        };

    }


    try {

        const response =
            await fetch(
                tinyFishFetchUrl,
                {
                    method:
                        "POST",

                    headers: {

                        "X-API-Key":
                            tinyFishApiKey,

                        "Content-Type":
                            "application/json"

                    },

                    body:
                        JSON.stringify({

                            urls: [
                                normalizedUrl.toString()
                            ],

                            format:
                                "markdown"

                        }),

                    signal:
                        AbortSignal.timeout(
                            45000
                        )
                }
            );


        const rawText =
            await response.text();


        if (!response.ok) {

            console.error(
                "TinyFish Fetch error:",
                response.status,
                rawText
            );


            return {
                success: false,

                text:
                    `Не удалось загрузить страницу: HTTP ${response.status}.`
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
                "TinyFish Fetch invalid JSON:",
                rawText
            );


            return {
                success: false,

                text:
                    "Сервис загрузки страницы вернул некорректные данные."
            };

        }


        const result =
            Array.isArray(
                data.results
            )
                ? data.results[0]
                : null;


        if (!result) {

            return {
                success: false,

                text:
                    "Не удалось получить содержимое страницы."
            };

        }


        const content =
            typeof result.text === "string"
                ? result.text.trim()
                : "";


        if (!content) {

            return {
                success: false,

                text:
                    "Страница загружена, но её содержимое получить не удалось."
            };

        }


        /*
         * Ограничиваем объём результата,
         * чтобы случайно не отправить огромную
         * страницу дальше по цепочке Jessica.
         *
         * Позже добавим нормальную работу
         * с большими документами и чанками.
         */
        const maxCharacters =
            30000;


        const truncated =
            content.length >
            maxCharacters;


        const finalContent =
            truncated
                ? content.slice(
                    0,
                    maxCharacters
                )
                : content;


        return {
            success: true,

            text:
                truncated
                    ? "Страница загружена. Содержимое было сокращено из-за большого объёма."
                    : "Страница успешно загружена.",

            data: {

                title:
                    typeof result.title === "string"
                        ? result.title
                        : "",

                url:
                    typeof result.url === "string"
                        ? result.url
                        : normalizedUrl.toString(),

                content:
                    finalContent,

                truncated

            }
        };


    } catch (error) {

        console.error(
            "Web Fetch Tool error:",
            error
        );


        return {
            success: false,

            text:
                "Jessica не смогла загрузить веб-страницу."
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
        "web_fetch",

    description:
        (
            "Открывает конкретную веб-страницу и получает её содержимое. " +
            "Используй, когда известен URL и для решения задачи нужно " +
            "прочитать саму страницу, а не только результаты поиска."
        ),

    arguments: {

        url:
            "Полный HTTP или HTTPS адрес страницы."

    },

    execute:
        executeWebFetch

});
