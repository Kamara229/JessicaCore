/*
 * =========================================================
 * JESSICA CORE BACKEND
 * =========================================================
 *
 * Главная точка входа backend Jessica.
 *
 * Архитектура:
 *
 * Android
 *   ↓
 * /api/solve
 *   ↓
 * Jessica Core
 *   ↓
 * Planner
 *   ↓
 * TaskRunner
 *   ↓
 * Tool Registry
 *   ↓
 * Answer Composer
 *   ↓
 * Validator
 *   ↓
 * Ответ
 */


/*
 * ВАЖНО:
 *
 * dotenv загружаем ДО модулей Jessica,
 * потому что инструменты читают process.env
 * во время своей инициализации.
 */
import "dotenv/config";

import express from "express";


/*
 * =========================================================
 * TOOLS INITIALIZATION
 * =========================================================
 *
 * При импорте initTools.js
 * все инструменты регистрируются
 * в Tool Registry.
 */
import "./tools/initTools.js";


/*
 * =========================================================
 * JESSICA CORE
 * =========================================================
 */

import {
    executeJessicaTask
} from "./core/jessicaCore.js";


/*
 * Прямой доступ к Tool Registry
 * нужен для служебных API:
 *
 * /api/search
 * /api/fetch
 * /api/health
 */
import {
    executeTool,
    getToolCount,
    listTools
} from "./tools/toolRegistry.js";


/*
 * =========================================================
 * EXPRESS
 * =========================================================
 */


const app =
    express();


app.use(
    express.json({
        limit:
            "1mb"
    })
);


/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


const port =
    Number(
        process.env.PORT
    ) || 3000;


const jessicaToken =
    process.env.JESSICA_APP_TOKEN || "";


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

            status:
                503,

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

            status:
                401,

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
 * AUTH MIDDLEWARE
 * =========================================================
 */


function requireJessicaAuthorization(
    req,
    res,
    next
) {

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


    next();

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
                "Jessica Core",

            version:
                "1.0",

            status:
                "running"

        });

    }
);


/*
 * =========================================================
 * HEALTH
 * =========================================================
 *
 * Здесь специально НЕ выводятся
 * значения API-ключей.
 *
 * Показывается только:
 * настроен сервис или нет.
 */


app.get(
    "/api/health",
    (
        req,
        res
    ) => {

        const tools =
            listTools();


        res.json({

            success: true,

            service:
                "Jessica Core",

            version:
                "1.0",

            architecture:
                "planner-runner-tools-composer-validator",

            configured: {

                appToken:
                    Boolean(
                        process.env.JESSICA_APP_TOKEN
                    ),

                groq:
                    Boolean(
                        process.env.GROQ_API_KEY
                    ),

                tinyFish:
                    Boolean(
                        process.env.TINYFISH_API_KEY
                    ),

                openAI:
                    Boolean(
                        process.env.OPENAI_API_KEY
                    )

            },

            tools: {

                count:
                    getToolCount(),

                registered:
                    tools.map(
                        tool =>
                            tool.name
                    )

            }

        });

    }
);


/*
 * =========================================================
 * DIRECT SEARCH API
 * =========================================================
 *
 * Сохраняем старый endpoint,
 * чтобы ничего не сломать.
 *
 * Но теперь он НЕ содержит
 * собственную реализацию TinyFish.
 *
 * Он использует Tool Registry.
 */


app.post(
    "/api/search",
    requireJessicaAuthorization,

    async (
        req,
        res
    ) => {

        try {

            const query =
                typeof req.body?.query === "string"
                    ? req.body.query.trim()
                    : "";


            if (!query) {

                return res
                    .status(
                        400
                    )
                    .json({

                        success: false,

                        text:
                            "Не указан поисковый запрос"

                    });

            }


            const result =
                await executeTool(
                    "web_search",
                    {
                        query
                    }
                );


            return res
                .status(
                    result.success
                        ? 200
                        : 502
                )
                .json(
                    result
                );


        } catch (error) {

            console.error(
                "/api/search error:",
                error
            );


            return res
                .status(
                    500
                )
                .json({

                    success: false,

                    text:
                        "Внутренняя ошибка интернет-поиска"

                });

        }

    }
);


/*
 * =========================================================
 * DIRECT FETCH API
 * =========================================================
 *
 * Аналогично /api/search:
 * endpoint сохраняется,
 * но реальная работа выполняется
 * зарегистрированным инструментом.
 */


app.post(
    "/api/fetch",
    requireJessicaAuthorization,

    async (
        req,
        res
    ) => {

        try {

            const url =
                typeof req.body?.url === "string"
                    ? req.body.url.trim()
                    : "";


            if (!url) {

                return res
                    .status(
                        400
                    )
                    .json({

                        success: false,

                        text:
                            "Не указан URL"

                    });

            }


            const result =
                await executeTool(
                    "web_fetch",
                    {
                        url
                    }
                );


            return res
                .status(
                    result.success
                        ? 200
                        : 502
                )
                .json(
                    result
                );


        } catch (error) {

            console.error(
                "/api/fetch error:",
                error
            );


            return res
                .status(
                    500
                )
                .json({

                    success: false,

                    text:
                        "Внутренняя ошибка загрузки страницы"

                });

        }

    }
);


/*
 * =========================================================
 * JESSICA SOLVE API
 * =========================================================
 *
 * Это теперь ГЛАВНАЯ точка новой архитектуры.
 *
 * Здесь больше НЕТ:
 *
 * taskNeedsInternet()
 * taskNeedsCurrentTime()
 * detectTimeZone()
 * словарей городов
 * ручного выбора инструмента
 * ручного маршрута Groq → TinyFish
 *
 * Всё решает Jessica Core.
 */


app.post(
    "/api/solve",
    requireJessicaAuthorization,

    async (
        req,
        res
    ) => {

        try {

            const task =
                typeof req.body?.task === "string"
                    ? req.body.task.trim()
                    : "";


            if (!task) {

                return res
                    .status(
                        400
                    )
                    .json({

                        success: false,

                        text:
                            "Задача не указана"

                    });

            }


            /*
             * Ограничение защищает backend
             * от случайной отправки огромных данных.
             *
             * Позже большие документы будут
             * обрабатываться отдельными инструментами.
             */
            if (
                task.length >
                20000
            ) {

                return res
                    .status(
                        413
                    )
                    .json({

                        success: false,

                        text:
                            "Задача слишком большого объёма"

                    });

            }


            console.log(
                "Jessica task:",
                task
            );


            /*
             * =================================================
             * НОВОЕ ЯДРО
             * =================================================
             */


            const result =
                await executeJessicaTask(
                    task
                );


            /*
             * Пользовательское уточнение —
             * это не внутренняя ошибка сервера.
             */
            if (
                result.needsClarification === true
            ) {

                return res
                    .status(
                        200
                    )
                    .json(
                        result
                    );

            }


            /*
             * Jessica смогла выполнить задачу.
             */
            if (
                result.success === true
            ) {

                return res
                    .status(
                        200
                    )
                    .json(
                        result
                    );

            }


            /*
             * Ошибка выполнения Jessica.
             *
             * Пока возвращаем 200,
             * потому что Android-клиент уже умеет
             * ориентироваться на поле success.
             *
             * Это позволяет не ломать
             * существующий JessicaAIEngine.kt.
             */
            return res
                .status(
                    200
                )
                .json(
                    result
                );


        } catch (error) {

            console.error(
                "/api/solve error:",
                error
            );


            return res
                .status(
                    500
                )
                .json({

                    success: false,

                    stage:
                        "server",

                    text:
                        "Внутренняя ошибка Jessica Core"

                });

        }

    }
);


/*
 * =========================================================
 * 404
 * =========================================================
 */


app.use(
    (
        req,
        res
    ) => {

        res
            .status(
                404
            )
            .json({

                success: false,

                text:
                    "Endpoint не найден"

            });

    }
);


/*
 * =========================================================
 * START SERVER
 * =========================================================
 */


app.listen(
    port,
    () => {

        const tools =
            listTools()
                .map(
                    tool =>
                        tool.name
                );


        console.log(
            `Jessica Core started on port ${port}`
        );


        console.log(
            `Jessica tools (${tools.length}):`,
            tools.join(", ")
        );

    }
);
