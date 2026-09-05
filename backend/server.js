import express from "express";
import OpenAI from "openai";
import dotenv from "dotenv";


dotenv.config();


const app = express();


app.use(
    express.json({
        limit: "1mb"
    })
);


/*
 * =========================================================
 * AI CLIENTS
 * =========================================================
 */


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
 * =========================================================
 * CONFIG
 * =========================================================
 */


const jessicaToken =
    process.env.JESSICA_APP_TOKEN || "";


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
 * TASK ROUTER
 * =========================================================
 */


/*
 * Определяем запрос точного времени.
 *
 * Такие задачи НЕ нужно отправлять
 * в интернет-поиск.
 */
function taskNeedsCurrentTime(
    task
) {

    const text =
        task.toLowerCase();


    const timeMarkers = [
        "который час",
        "сколько времени",
        "текущее время",
        "время сейчас",
        "сейчас времени",
        "what time",
        "current time"
    ];


    return timeMarkers.some(
        marker =>
            text.includes(
                marker
            )
    );

}


/*
 * Определяем запрос текущей даты.
 */
function taskNeedsCurrentDate(
    task
) {

    const text =
        task.toLowerCase();


    const dateMarkers = [
        "какое сегодня число",
        "какая сегодня дата",
        "дата сегодня",
        "сегодняшняя дата",
        "какой сегодня день",
        "current date",
        "today's date"
    ];


    return dateMarkers.some(
        marker =>
            text.includes(
                marker
            )
    );

}


/*
 * Определяем часовой пояс по тексту.
 *
 * Пока это базовый словарь.
 * Его потом можно расширять.
 */
function detectTimeZone(
    task
) {

    const text =
        task.toLowerCase();


    const zones = [

        {
            markers: [
                "ростов-на-дону",
                "ростове-на-дону",
                "ростове",
                "москве",
                "москва",
                "санкт-петербург",
                "петербург",
                "спб",
                "сочи",
                "краснодар"
            ],
            zone:
                "Europe/Moscow"
        },

        {
            markers: [
                "калининград"
            ],
            zone:
                "Europe/Kaliningrad"
        },

        {
            markers: [
                "самара"
            ],
            zone:
                "Europe/Samara"
        },

        {
            markers: [
                "екатеринбург",
                "екб"
            ],
            zone:
                "Asia/Yekaterinburg"
        },

        {
            markers: [
                "омск"
            ],
            zone:
                "Asia/Omsk"
        },

        {
            markers: [
                "новосибирск",
                "красноярск"
            ],
            zone:
                "Asia/Krasnoyarsk"
        },

        {
            markers: [
                "иркутск"
            ],
            zone:
                "Asia/Irkutsk"
        },

        {
            markers: [
                "якутск"
            ],
            zone:
                "Asia/Yakutsk"
        },

        {
            markers: [
                "владивосток"
            ],
            zone:
                "Asia/Vladivostok"
        },

        {
            markers: [
                "магадан"
            ],
            zone:
                "Asia/Magadan"
        },

        {
            markers: [
                "камчатка",
                "петропавловск-камчатский"
            ],
            zone:
                "Asia/Kamchatka"
        },

        {
            markers: [
                "лондон"
            ],
            zone:
                "Europe/London"
        },

        {
            markers: [
                "амстердам"
            ],
            zone:
                "Europe/Amsterdam"
        },

        {
            markers: [
                "берлин"
            ],
            zone:
                "Europe/Berlin"
        },

        {
            markers: [
                "париж"
            ],
            zone:
                "Europe/Paris"
        },

        {
            markers: [
                "нью-йорк",
                "нью йорк"
            ],
            zone:
                "America/New_York"
        },

        {
            markers: [
                "лос-анджелес",
                "лос анджелес"
            ],
            zone:
                "America/Los_Angeles"
        },

        {
            markers: [
                "дубай"
            ],
            zone:
                "Asia/Dubai"
        },

        {
            markers: [
                "токио"
            ],
            zone:
                "Asia/Tokyo"
        }

    ];


    for (
        const item
        of zones
    ) {

        if (
            item.markers.some(
                marker =>
                    text.includes(
                        marker
                    )
            )
        ) {

            return item.zone;

        }

    }


    /*
     * Jessica сейчас используется
     * преимущественно в московском часовом поясе.
     *
     * Для запроса без города используем UTC,
     * чтобы не выдавать ложное локальное время.
     */
    return null;

}


/*
 * Решение задачи времени без AI и веба.
 */
function solveCurrentTime(
    task
) {

    const timeZone =
        detectTimeZone(
            task
        );


    if (!timeZone) {

        return {
            success: false,
            text:
                "Не удалось определить город или часовой пояс."
        };

    }


    try {

        const now =
            new Date();


        const time =
            new Intl.DateTimeFormat(
                "ru-RU",
                {
                    timeZone,
                    hour:
                        "2-digit",
                    minute:
                        "2-digit",
                    second:
                        "2-digit",
                    hour12:
                        false
                }
            ).format(
                now
            );


        const date =
            new Intl.DateTimeFormat(
                "ru-RU",
                {
                    timeZone,
                    day:
                        "2-digit",
                    month:
                        "long",
                    year:
                        "numeric"
                }
            ).format(
                now
            );


        return {
            success: true,
            text:
                `Сейчас ${time}. Дата: ${date}.`,
            timeZone
        };


    } catch (error) {

        console.error(
            "Time tool error:",
            error
        );


        return {
            success: false,
            text:
                "Не удалось определить текущее время."
        };

    }

}


/*
 * Решение задачи даты без AI и веба.
 */
function solveCurrentDate(
    task
) {

    const timeZone =
        detectTimeZone(
            task
        ) ||
        "Europe/Moscow";


    try {

        const now =
            new Date();


        const date =
            new Intl.DateTimeFormat(
                "ru-RU",
                {
                    timeZone,
                    weekday:
                        "long",
                    day:
                        "2-digit",
                    month:
                        "long",
                    year:
                        "numeric"
                }
            ).format(
                now
            );


        return {
            success: true,
            text:
                `Сегодня ${date}.`,
            timeZone
        };


    } catch (error) {

        console.error(
            "Date tool error:",
            error
        );


        return {
            success: false,
            text:
                "Не удалось определить текущую дату."
        };

    }

}


/*
 * Определяем необходимость свежих
 * данных из интернета.
 */
function taskNeedsInternet(
    task
) {

    /*
     * Время и дата имеют собственные инструменты.
     */
    if (
        taskNeedsCurrentTime(
            task
        ) ||
        taskNeedsCurrentDate(
            task
        )
    ) {

        return false;

    }


    const text =
        task.toLowerCase();


    const markers = [

        "сегодня",
        "сейчас",
        "актуальн",
        "последн",
        "свеж",
        "новост",

        "найди",
        "найти",
        "поищи",
        "поиск",

        "в интернете",
        "на сайте",
        "сайт",

        "цена",
        "стоимость",
        "сколько стоит",

        "курс валют",
        "курс доллара",
        "курс евро",

        "погода",
        "температура",
        "осадки",

        "расписание",
        "рейс",
        "поезд",
        "самолет",

        "где купить",
        "в наличии",

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

                    `РЕЗУЛЬТАТЫ ИНТЕРНЕТ-ПОИСКА:\n` +
                    `${webContext}\n\n` +

                    `Используй эти данные для ответа. ` +
                    `Оценивай актуальность и непротиворечивость источников. ` +
                    `Не придумывай отсутствующие сведения. ` +
                    `Если данные противоречат друг другу, не выбирай случайное значение. ` +
                    `Объясни неопределённость кратко.`
                );

        }


        const response =
            await groq.responses.create({

                model:
                    "openai/gpt-oss-20b",

                instructions:
                    (
                        "Ты — основной AI-движок системы Jessica Core. " +

                        "Твоя задача — давать пользователю готовый полезный ответ, " +
                        "а не описывать процесс рассуждений. " +

                        "Отвечай на языке пользователя. " +

                        "Не используй Markdown-таблицы с символами |, " +
                        "потому что Android-интерфейс Jessica пока не отображает их корректно. " +

                        "Для простого вопроса отвечай кратко. " +
                        "Для сложной задачи можешь отвечать подробно. " +

                        "Если Jessica передала результаты интернет-поиска, " +
                        "используй их только как источники данных. " +

                        "Не перечисляй все найденные источники без необходимости. " +
                        "Сначала дай пользователю прямой ответ. " +

                        "Не утверждай, что выполнила действие во внешней системе, " +
                        "если оно фактически не было выполнено. " +

                        "Не выдумывай актуальные факты. " +

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

                    `РЕЗУЛЬТАТЫ ИНТЕРНЕТ-ПОИСКА:\n` +
                    `${webContext}`
                );

        }


        const response =
            await openai.responses.create({

                model:
                    "gpt-5.6-sol",

                instructions:
                    (
                        "Ты — резервный AI-движок Jessica Core. " +
                        "Дай готовый точный ответ на языке пользователя. " +
                        "Не используй Markdown-таблицы. " +
                        "Не выдумывай данные. " +
                        "Не утверждай, что выполнила внешнее действие, " +
                        "если оно фактически не выполнялось."
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
                openai !== null,

            routerVersion:
                "0.2"

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
             * =================================================
             * 1. DIRECT TOOLS
             * =================================================
             *
             * Сначала Jessica пытается решить задачу
             * собственным специализированным инструментом.
             */


            if (
                taskNeedsCurrentTime(
                    task
                )
            ) {

                const timeResult =
                    solveCurrentTime(
                        task
                    );


                if (
                    timeResult.success
                ) {

                    return res.json({

                        success: true,

                        text:
                            timeResult.text,

                        engine:
                            "jessica-time",

                        webUsed:
                            false,

                        paidAIUsed:
                            false

                    });

                }

            }


            if (
                taskNeedsCurrentDate(
                    task
                )
            ) {

                const dateResult =
                    solveCurrentDate(
                        task
                    );


                if (
                    dateResult.success
                ) {

                    return res.json({

                        success: true,

                        text:
                            dateResult.text,

                        engine:
                            "jessica-date",

                        webUsed:
                            false,

                        paidAIUsed:
                            false

                    });

                }

            }


            /*
             * =================================================
             * 2. ROUTING
             * =================================================
             */


            const needsInternet =
                taskNeedsInternet(
                    task
                );


            let webContext = "";


            let webUsed =
                false;


            /*
             * =================================================
             * 3. FREE WEB
             * =================================================
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
             * =================================================
             * 4. FREE AI
             * =================================================
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

                    text:
                        groqResult.text,

                    engine:
                        "groq",

                    webUsed,

                    paidAIUsed:
                        false

                });

            }


            /*
             * =================================================
             * 5. PAID AI FALLBACK
             * =================================================
             *
             * OpenAI вызывается только тогда,
             * когда Groq действительно не смог
             * выполнить задачу.
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

                    text:
                        openAIResult.text,

                    engine:
                        "openai",

                    webUsed,

                    paidAIUsed:
                        true

                });

            }


            /*
             * Все способы исчерпаны.
             */
            return res
                .status(502)
                .json({

                    success: false,

                    text:
                        (
                            "Jessica не смогла выполнить задачу. " +
                            `Groq: ${groqResult.text} ` +
                            `OpenAI: ${openAIResult.text}`
                        ),

                    webUsed,

                    paidAIUsed:
                        false

                });


        } catch (error) {

            console.error(
                "Solve endpoint error:",
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
 * START SERVER
 * =========================================================
 */


const port =
    process.env.PORT || 3000;


app.listen(
    port,
    () => {

        console.log(
            `Jessica Backend started on port ${port}`
        );

    }
);
