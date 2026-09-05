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
 * Используется как резервный AI-движок.
 */
const openai =
    process.env.OPENAI_API_KEY
        ? new OpenAI({
            apiKey:
                process.env.OPENAI_API_KEY
        })
        : null;


/*
 * Токен Android-приложения Jessica.
 */
const jessicaToken =
    process.env.JESSICA_APP_TOKEN || "";


/*
 * TinyFish API.
 */
const tinyFishApiKey =
    process.env.TINYFISH_API_KEY || "";


const tinyFishSearchUrl =
    "https://api.search.tinyfish.ai";


const tinyFishFetchUrl =
    "https://api.fetch.tinyfish.ai";


/*
 * Проверка авторизации Jessica.
 */
function checkJessicaAuthorization(req) {

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
 * Бесплатный интернет-поиск TinyFish.
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
                    method: "GET",

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
 * Бесплатное чтение страницы TinyFish Fetch.
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
                    method: "POST",

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
 * Формирование читаемого результата поиска.
 */
function formatSearchResults(
    results
) {

    if (
        !results ||
        results.length === 0
    ) {

        return "Ничего не найдено.";

    }


    return results
        .slice(
            0,
            10
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
                    "";


                const url =
                    item.url ||
                    "";


                return (
                    `${index + 1}. ${title}\n` +
                    `${snippet}\n` +
                    `${url}`
                );

            }
        )
        .join(
            "\n\n"
        );

}


/*
 * Главная страница backend.
 */
app.get(
    "/",
    (req, res) => {

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
 * Проверка состояния backend.
 */
app.get(
    "/api/health",
    (req, res) => {

        res.json({
            success: true,
            service:
                "Jessica Backend",
            status:
                "ok",

            aiConfigured:
                openai !== null,

            appAuthConfigured:
                jessicaToken.length > 0,

            tinyFishConfigured:
                tinyFishApiKey.length > 0
        });

    }
);


/*
 * Отдельный endpoint интернет-поиска.
 *
 * Пока нужен для тестирования.
 * Позже Jessica будет вызывать его
 * самостоятельно через Planner.
 */
app.post(
    "/api/search",
    async (req, res) => {

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
                .json(result);

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
 * Endpoint чтения интернет-страницы.
 */
app.post(
    "/api/fetch",
    async (req, res) => {

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
                .json(result);

        }


        return res.json(result);

    }
);


/*
 * Выполнение задачи Jessica.
 *
 * Пока сохраняем существующее
 * поведение через OpenAI.
 *
 * На следующем этапе сюда
 * добавим Planner:
 *
 * Jessica
 * → свои возможности
 * → память
 * → TinyFish
 * → бесплатный AI
 * → OpenAI только при необходимости.
 */
app.post(
    "/api/solve",
    async (req, res) => {

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
             * Пока OpenAI остаётся
             * резервным AI Engine.
             */
            if (!openai) {

                return res
                    .status(503)
                    .json({
                        success: false,
                        text:
                            "AI Engine не настроен"
                    });

            }


            const response =
                await openai.responses.create({

                    model:
                        "gpt-5.6-sol",

                    instructions:
                        "Ты являешься резервным AI-ядром системы Jessica Core. " +
                        "Выполняй задачу пользователя точно, полезно и по существу. " +
                        "Отвечай на языке, на котором сформулирована задача. " +
                        "Не утверждай, что выполнил действия во внешних системах, " +
                        "если фактически у тебя нет соответствующего инструмента.",

                    input:
                        task

                });


            const answer =
                response.output_text
                    ?.trim();


            if (!answer) {

                return res
                    .status(502)
                    .json({
                        success: false,
                        text:
                            "Модель вернула пустой ответ"
                    });

            }


            return res.json({
                success: true,
                source:
                    "openai",
                text:
                    answer
            });


        } catch (error) {

            console.error(
                "Jessica AI error:",
                error
            );


            const status =
                error?.status;


            if (status === 429) {

                return res
                    .status(429)
                    .json({
                        success: false,
                        text:
                            "Баланс OpenAI API исчерпан. Пополните баланс и повторите задачу."
                    });

            }


            if (status === 401) {

                return res
                    .status(502)
                    .json({
                        success: false,
                        text:
                            "Ошибка авторизации OpenAI API"
                    });

            }


            if (status === 403) {

                return res
                    .status(502)
                    .json({
                        success: false,
                        text:
                            "OpenAI API не разрешил доступ к выбранной модели"
                    });

            }


            if (
                status === 500 ||
                status === 502 ||
                status === 503 ||
                status === 504
            ) {

                return res
                    .status(503)
                    .json({
                        success: false,
                        text:
                            "OpenAI временно недоступен. Попробуйте повторить задачу позже."
                    });

            }


            return res
                .status(500)
                .json({
                    success: false,
                    text:
                        "Внутренняя ошибка AI-сервера"
                });

        }

    }
);


/*
 * Render автоматически передаёт PORT.
 */
const port =
    Number(
        process.env.PORT
    ) || 3000;


/*
 * 0.0.0.0 нужен для Render.
 */
app.listen(
    port,
    "0.0.0.0",
    () => {

        console.log(
            `Jessica Backend запущен на порту ${port}`
        );

    }
);
