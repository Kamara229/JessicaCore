import express from "express";
import OpenAI from "openai";
import dotenv from "dotenv";


dotenv.config();


const app =
    express();


app.use(
    express.json({
        limit: "1mb"
    })
);


/*
 * OpenAI.
 * Платный резервный AI.
 */
const openai =
    process.env.OPENAI_API_KEY
        ? new OpenAI({
            apiKey:
                process.env.OPENAI_API_KEY
        })
        : null;


/*
 * Groq.
 * Основной бесплатный AI Jessica.
 *
 * Groq совместим с OpenAI Responses API.
 */
const groq =
    process.env.GROQ_API_KEY
        ? new OpenAI({
            apiKey:
                process.env.GROQ_API_KEY,

            baseURL:
                "https://api.groq.com/openai/v1"
        })
        : null;


/*
 * Авторизация Android-приложения.
 */
const jessicaToken =
    process.env.JESSICA_APP_TOKEN || "";


/*
 * TinyFish.
 * Бесплатный интернет-поиск.
 */
const tinyFishApiKey =
    process.env.TINYFISH_API_KEY || "";


const tinyFishSearchUrl =
    "https://api.search.tinyfish.ai";


const tinyFishFetchUrl =
    "https://api.fetch.tinyfish.ai";


/*
 * =========================================================
 * AUTH
 * =========================================================
 */


function checkJessicaAuthorization(
    req
) {

    if (!jessicaToken) {

        return {
            success: false,
            status: 503,
            text:
                "Авторизация Jessica не настроена на сервере"
        };

    }


    const appToken =
        req.get(
            "X-Jessica-Token"
        ) || "";


    if (
        appToken !==
        jessicaToken
    ) {

        return {
            success: false,
            status: 401,
            text:
                "Неавторизованный запрос"
        };

    }


    return {
        success: true
    };

}


/*
 * =========================================================
 * INTERNET DETECTION
 * =========================================================
 */


/*
 * Простая первая версия определения,
 * нужны ли задаче свежие данные из интернета.
 *
 * Позже это решение будет принимать
 * Planner / Development Core.
 */
function taskNeedsInternet(
    task
) {

    const text =
        task.toLowerCase();


    const markers = [

        "сегодня",
        "сейчас",
        "актуальн",
        "последн",
        "новост",
        "найди",
        "найти",
        "поиск",
        "поищи",
        "интернет",
        "в интернете",
        "на сайте",
        "сайт",
        "цена",
        "стоимость",
        "курс",
        "погода",
        "расписание",
        "где купить",
        "в наличии",
        "2026",
        "latest",
        "today",
        "current",
        "news",
        "search",
        "find online",
        "website"

    ];


    return markers.some(
        marker =>
            text.includes(
                marker
            )
    );

}


/*
 * =========================================================
 * TINYFISH SEARCH
 * =========================================================
 */


async function searchWeb(
    query
) {

    if (!tinyFishApiKey) {

        return {
            success: false,
            text:
                "TinyFish Search не настроен",
            results: []
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
                    `Ошибка TinyFish Search: HTTP ${response.status}`,
                results: []
            };

        }


        let data;


        try {

            data =
                JSON.parse(
                    rawText
                );

        } catch {

            return {
                success: false,
                text:
                    "TinyFish Search вернул некорректный ответ",
                results: []
            };

        }


        const results =
            Array.isArray(
                data.results
            )
                ? data.results
                : [];


        return {
            success: true,
            text:
                `Найдено результатов: ${results.length}`,
            results
        };


    } catch (error) {

        console.error(
            "TinyFish Search exception:",
            error
        );


        return {
            success: false,
            text:
                "Не удалось выполнить интернет-поиск",
            results: []
        };

    }

}


/*
 * =========================================================
 * TINYFISH FETCH
 * =========================================================
 */


async function fetchWebPage(
    url
) {

    if (!tinyFishApiKey) {

        return {
            success: false,
            text:
                "TinyFish Fetch не настроен"
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
                                url
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
                    `Ошибка TinyFish Fetch: HTTP ${response.status}`
            };

        }


        let data;


        try {

            data =
                JSON.parse(
                    rawText
                );

        } catch {

            return {
                success: false,
                text:
                    "TinyFish Fetch вернул некорректный ответ"
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
                    "TinyFish не смог получить содержимое страницы"
            };

        }


        return {
            success: true,

            title:
                result.title || "",

            url:
                result.url || url,

            text:
                result.text || ""
        };


    } catch (error) {

        console.error(
            "TinyFish Fetch exception:",
            error
        );


        return {
            success: false,
            text:
                "Не удалось загрузить страницу"
        };

    }

}


/*
 * =========================================================
 * SEARCH FORMATTING
 * =========================================================
 */


function formatSearchResults(
    results
) {

    if (
        !results ||
        results.length === 0
    ) {

        return "";

    }


    return results
        .slice(
            0,
            8
        )
        .map(
            (
                item,
                index
            ) => {

                const title =
                    item.title ||
                    "Без названия";


                const snippet =
                    item.snippet ||
                    item.description ||
                    "";


                const url =
                    item.url ||
                    "";


                return (
                    `[Источник ${index + 1}]\n` +
                    `Название: ${title}\n` +
                    `Фрагмент: ${snippet}\n` +
                    `URL: ${url}`
                );

            }
        )
        .join(
            "\n\n"
        );

}


/*
 * =========================================================
 * GROQ
 * =========================================================
 */


async function solveWithGroq(
    task,
    webContext = ""
) {

    if (!groq) {

        return {
            success: false,
            text:
                "Groq не настроен"
        };

    }


    try {

        let input =
            task;


        if (webContext) {

            input =
                (
                    `ЗАДАЧА ПОЛЬЗОВАТЕЛЯ:\n` +
                    `${task}\n\n` +

                    `ДАННЫЕ ИЗ ИНТЕРНЕТ-ПОИСКА:\n` +
                    `${webContext}\n\n` +

                    `Используй результаты поиска только как источники данных. ` +
                    `Не придумывай информацию, которой в них нет. ` +
                    `Если источники противоречат друг другу, укажи это.`
                );

        }


        const response =
            await groq.responses.create({

                model:
                    "openai/gpt-oss-20b",

                instructions:
                    (
                        "Ты являешься основным бесплатным AI-движком системы Jessica Core. " +
                        "Решай задачу самостоятельно, точно и полезно. " +
                        "Отвечай на языке пользователя. " +
                        "Используй предоставленные Jessica данные и результаты интернет-поиска. " +
                        "Не утверждай, что совершила внешнее действие, если фактически оно не было выполнено. " +
                        "Если информации недостаточно, прямо скажи об этом."
                    ),

                input,

                reasoning: {
                    effort:
                        "medium"
                }

            });


        const answer =
            response.output_text
                ?.trim();


        if (!answer) {

            return {
                success: false,
                text:
                    "Groq вернул пустой ответ"
            };

        }


        return {
            success: true,
            text:
                answer
        };


    } catch (error) {

        console.error(
            "Groq error:",
            error
        );


        return {
            success: false,

            status:
                error?.status || 0,

            text:
                "Groq не смог выполнить задачу"
        };

    }

}


/*
 * =========================================================
 * OPENAI FALLBACK
 * =========================================================
 */


async function solveWithOpenAI(
    task,
    webContext = ""
) {

    if (!openai) {

        return {
            success: false,
            text:
                "OpenAI не настроен"
        };

    }


    try {

        let input =
            task;


        if (webContext) {

            input =
                (
                    `ЗАДАЧА ПОЛЬЗОВАТЕЛЯ:\n` +
                    `${task}\n\n` +

                    `ДАННЫЕ ИЗ ИНТЕРНЕТ-ПОИСКА:\n` +
                    `${webContext}`
                );

        }


        const response =
            await openai.responses.create({

                model:
                    "gpt-5.6-sol",

                instructions:
                    (
                        "Ты являешься резервным AI-движком системы Jessica Core. " +
                        "Используйся только тогда, когда бесплатные возможности Jessica не справились. " +
                        "Выполняй задачу точно, полезно и по существу. " +
                        "Отвечай на языке пользователя. " +
                        "Не утверждай, что совершила внешние действия, если они фактически не выполнялись."
                    ),

                input

            });


        const answer =
            response.output_text
                ?.trim();


        if (!answer) {

            return {
                success: false,
                text:
                    "OpenAI вернул пустой ответ"
            };

        }


        return {
            success: true,
            text:
                answer
        };


    } catch (error) {

        console.error(
            "OpenAI error:",
            error
        );


        if (
            error?.status === 429
        ) {

            return {
                success: false,
                status: 429,
                text:
                    "Баланс OpenAI API исчерпан."
            };

        }


        return {
            success: false,

            status:
                error?.status || 0,

            text:
                "OpenAI не смог выполнить задачу"
        };

    }

}


/*
 * =========================================================
 * ROOT
 * =========================================================
 */


app.get(
    "/",
    (
        req,
        res
    ) => {

        res.json({

            success: true,

            service:
                "Jessica Backend",

            status:
                "online"

        });

    }
);


/*
 * =========================================================
 * HEALTH
 * =========================================================
 */


app.get(
    "/api/health",
    (
        req,
        res
    ) => {

        res.json({

            success: true,

            service:
                "Jessica Backend",

            status:
                "ok",

            appAuthConfigured:
                jessicaToken.length > 0,

            tinyFishConfigured:
                tinyFishApiKey.length > 0,

            groqConfigured:
                groq !== null,

            openAIConfigured:
                openai !== null

        });

    }
);


/*
 * =========================================================
 * SEARCH
 * =========================================================
 */


app.post(
    "/api/search",
    async (
        req,
        res
    ) => {

        const auth =
            checkJessicaAuthorization(
                req
            );


        if (!auth.success) {

            return res
                .status(
                    auth.status
                )
                .json({

                    success: false,

                    text:
                        auth.text

                });

        }


        const query =
            typeof req.body?.query ===
            "string"
                ? req.body.query.trim()
                : "";


        if (!query) {

            return res
                .status(400)
                .json({

                    success: false,

                    text:
                        "Поисковый запрос не указан"

                });

        }


        const result =
            await searchWeb(
                query
            );


        if (!result.success) {

            return res
                .status(502)
                .json(
                    result
                );

        }


        return res.json({

            success: true,

            query,

            text:
                formatSearchResults(
                    result.results
                ),

            results:
                result.results

        });

    }
);


/*
 * =========================================================
 * FETCH
 * =========================================================
 */


app.post(
    "/api/fetch",
    async (
        req,
        res
    ) => {

        const auth =
            checkJessicaAuthorization(
                req
            );


        if (!auth.success) {

            return res
                .status(
                    auth.status
                )
                .json({

                    success: false,

                    text:
                        auth.text

                });

        }


        const url =
            typeof req.body?.url ===
            "string"
                ? req.body.url.trim()
                : "";


        if (!url) {

            return res
                .status(400)
                .json({

                    success: false,

                    text:
                        "URL не указан"

                });

        }


        const result =
            await fetchWebPage(
                url
            );


        if (!result.success) {

            return res
                .status(502)
                .json(
                    result
                );

        }


        return res.json(
            result
        );

    }
);


/*
 * =========================================================
 * SOLVE
 * =========================================================
 */


app.post(
    "/api/solve",
    async (
        req,
        res
    ) => {

        try {

            const auth =
                checkJessicaAuthorization(
                    req
                );


            if (!auth.success) {

                return res
                    .status(
                        auth.status
                    )
                    .json({

                        success: false,

                        text:
                            auth.text

                    });

            }


            const task =
                typeof req.body?.task ===
                "string"
                    ? req.body.task.trim()
                    : "";


            if (!task) {

                return res
                    .status(400)
                    .json({

                        success: false,

                        text:
                            "Задача не указана"

                    });

            }


            /*
             * 1.
             * Определяем, нужен ли интернет.
             */
            const needsInternet =
                taskNeedsInternet(
                    task
                );


            let webContext = "";

            let webUsed =
                false;


            /*
             * 2.
             * Если нужен интернет —
             * сначала бесплатный TinyFish.
             */
            if (
                needsInternet &&
                tinyFishApiKey
            ) {

                const searchResult =
                    await searchWeb(
                        task
                    );


                if (
                    searchResult.success &&
                    searchResult.results.length > 0
                ) {

                    webContext =
                        formatSearchResults(
                            searchResult.results
                        );


                    webUsed =
                        true;

                }

            }


            /*
             * 3.
             * Основной бесплатный AI —
             * Groq.
             */
            const groqResult =
                await solveWithGroq(
                    task,
                    webContext
                );


            if (
                groqResult.success
            ) {

                return res.json({

                    success: true,

                    source:
                        webUsed
                            ? "tinyfish+groq"
                            : "groq",

                    internetUsed:
                        webUsed,

                    openAIUsed:
                        false,

                    text:
                        groqResult.text

                });

            }


            /*
             * 4.
             * Только если Groq не справился —
             * используем OpenAI.
             */
            const openAIResult =
                await solveWithOpenAI(
                    task,
                    webContext
                );


            if (
                openAIResult.success
            ) {

                return res.json({

                    success: true,

                    source:
                        webUsed
                            ? "tinyfish+openai"
                            : "openai",

                    internetUsed:
                        webUsed,

                    openAIUsed:
                        true,

                    text:
                        openAIResult.text

                });

            }


            /*
             * 5.
             * Если бесплатный AI не сработал,
             * а OpenAI недоступен из-за баланса.
             */
            if (
                openAIResult.status === 429
            ) {

                return res
                    .status(503)
                    .json({

                        success: false,

                        source:
                            "none",

                        internetUsed:
                            webUsed,

                        openAIUsed:
                            true,

                        text:
                            (
                                "Jessica не смогла завершить задачу бесплатными средствами. " +
                                "Резервный OpenAI сейчас недоступен из-за отсутствия API-баланса."
                            )

                    });

            }


            /*
             * 6.
             * Общая ошибка.
             */
            return res
                .status(503)
                .json({

                    success: false,

                    source:
                        "none",

                    internetUsed:
                        webUsed,

                    openAIUsed:
                        false,

                    text:
                        "Jessica не смогла выполнить задачу доступными AI-движками."

                });


        } catch (error) {

            console.error(
                "Jessica solve error:",
                error
            );


            return res
                .status(500)
                .json({

                    success: false,

                    text:
                        "Внутренняя ошибка Jessica Backend"

                });

        }

    }
);


/*
 * =========================================================
 * SERVER
 * =========================================================
 */


const port =
    Number(
        process.env.PORT
    ) || 3000;


app.listen(
    port,
    "0.0.0.0",
    () => {

        console.log(
            `Jessica Backend запущен на порту ${port}`
        );

    }
);
