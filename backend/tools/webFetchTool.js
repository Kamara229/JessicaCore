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
 * ВАЖНО:
 *
 * web_fetch НЕ формирует конечный ответ пользователю.
 *
 * Он только возвращает содержимое страницы,
 * которое затем анализирует Jessica.
 *
 * Поэтому при успехе:
 *
 * text = ""
 * data = содержимое страницы
 */


/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


const tinyFishApiKey =
    process.env.TINYFISH_API_KEY || "";


const tinyFishFetchUrl =
    "https://api.fetch.tinyfish.ai";


const MAX_CHARACTERS =
    30000;


/*
 * =========================================================
 * URL VALIDATION
 * =========================================================
 */


function normalizeHttpUrl(
    value
) {

    if (
        typeof value !== "string" ||
        !value.trim()
    ) {

        return null;

    }


    try {

        const url =
            new URL(
                value.trim()
            );


        if (
            url.protocol !== "http:" &&
            url.protocol !== "https:"
        ) {

            return null;

        }


        return url.toString();


    } catch {

        return null;

    }

}


/*
 * =========================================================
 * FETCH PAGE
 * =========================================================
 */


async function executeWebFetch(
    args
) {

    const rawUrl =
        typeof args?.url === "string"
            ? args.url.trim()
            : "";


    /*
     * -----------------------------------------------------
     * INPUT VALIDATION
     * -----------------------------------------------------
     */


    if (!rawUrl) {

        return {
            success: false,

            needsClarification:
                true,

            text:
                "Не указан адрес страницы.",

            data:
                null
        };

    }


    const pageUrl =
        normalizeHttpUrl(
            rawUrl
        );


    if (!pageUrl) {

        return {
            success: false,

            text:
                "Указан некорректный HTTP или HTTPS адрес страницы.",

            data:
                null
        };

    }


    if (!tinyFishApiKey) {

        return {
            success: false,

            text:
                "Загрузка веб-страниц Jessica не настроена.",

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
                                pageUrl
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


        /*
         * -------------------------------------------------
         * HTTP ERROR
         * -------------------------------------------------
         */


        if (!response.ok) {

            console.error(
                "TinyFish Fetch error:",
                response.status,
                rawText
            );


            return {
                success: false,

                text:
                    `Не удалось загрузить страницу: HTTP ${response.status}.`,

                data:
                    null
            };

        }


        /*
         * -------------------------------------------------
         * PARSE RESPONSE
         * -------------------------------------------------
         */


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
                    "Сервис загрузки страницы вернул некорректные данные.",

                data:
                    null
            };

        }


        const result =
            Array.isArray(
                data?.results
            )
                ? data.results[0]
                : null;


        if (!result) {

            return {
                success: false,

                text:
                    "Не удалось получить содержимое страницы.",

                data:
                    null
            };

        }


        /*
         * -------------------------------------------------
         * CONTENT
         * -------------------------------------------------
         */


        const content =
            typeof result.text === "string"
                ? result.text.trim()
                : "";


        if (!content) {

            return {
                success: false,

                text:
                    "Страница загружена, но её содержимое получить не удалось.",

                data:
                    null
            };

        }


        const truncated =
            content.length >
            MAX_CHARACTERS;


        const finalContent =
            truncated
                ? content.slice(
                    0,
                    MAX_CHARACTERS
                )
                : content;


        /*
         * -------------------------------------------------
         * SUCCESS
         * -------------------------------------------------
         *
         * text специально пустой.
         *
         * Содержимое страницы —
         * это данные для дальнейшего анализа,
         * а не готовый ответ пользователю.
         */


        return {
            success: true,

            text:
                "",

            data: {

                title:
                    typeof result.title === "string"
                        ? result.title.trim()
                        : "",

                url:
                    typeof result.url === "string" &&
                    result.url.trim()
                        ? result.url.trim()
                        : pageUrl,

                content:
                    finalContent,

                truncated,

                originalLength:
                    content.length,

                returnedLength:
                    finalContent.length

            }
        };


    } catch (error) {

        console.error(
            "Web Fetch Tool exception:",
            error
        );


        if (
            error?.name ===
            "TimeoutError"
        ) {

            return {
                success: false,

                text:
                    "Загрузка страницы превысила допустимое время ожидания.",

                data:
                    null
            };

        }


        return {
            success: false,

            text:
                "Jessica не смогла загрузить веб-страницу.",

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
        "web_fetch",

    description:
        (
            "Открывает конкретную HTTP или HTTPS страницу " +
            "и возвращает её содержимое для дальнейшего анализа Jessica. " +

            "Используй, когда URL уже известен и для решения задачи " +
            "нужно прочитать сам источник, проверить детали " +
            "или получить данные, которых недостаточно в поисковом результате. " +

            "Инструмент не формирует конечный ответ пользователю."
        ),

    arguments: {

        url:
            (
                "Полный HTTP или HTTPS адрес страницы, " +
                "которую необходимо открыть."
            )

    },

    execute:
        executeWebFetch

});
