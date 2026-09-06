import OpenAI from "openai";

import {
    listTools
} from "../tools/toolRegistry.js";


/*
 * =========================================================
 * JESSICA TASK PLANNER
 * =========================================================
 *
 * Planner:
 *
 * 1. понимает цель пользователя;
 * 2. определяет, нужны ли инструменты;
 * 3. выбирает только зарегистрированные tools;
 * 4. создаёт один или несколько шагов;
 * 5. может передавать результаты предыдущих шагов
 *    следующим шагам через ссылки:
 *
 * {
 *     "$from": "search",
 *     "path": "data.results.0.url"
 * }
 *
 * Planner НЕ решает задачу пользователя.
 */


/*
 * =========================================================
 * AI CLIENT
 * =========================================================
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


const MAX_ATTEMPTS =
    3;


const MAX_STEPS =
    15;


const RETRY_DELAY_MS =
    1200;


/*
 * =========================================================
 * HELPERS
 * =========================================================
 */


function sleep(
    milliseconds
) {

    return new Promise(
        resolve =>
            setTimeout(
                resolve,
                milliseconds
            )
    );

}


/*
 * =========================================================
 * TOOLS DESCRIPTION
 * =========================================================
 */


function buildToolsDescription() {

    const tools =
        listTools();


    if (
        tools.length === 0
    ) {

        return (
            "Сейчас зарегистрированных инструментов нет. " +
            "Можно создать только план без инструментов."
        );

    }


    return tools
        .map(
            tool =>
                JSON.stringify(
                    {
                        name:
                            tool.name,

                        description:
                            tool.description,

                        arguments:
                            tool.arguments || {}
                    },
                    null,
                    2
                )
        )
        .join(
            "\n\n"
        );

}


/*
 * =========================================================
 * JSON CLEANUP
 * =========================================================
 */


function cleanJsonText(
    text
) {

    if (
        typeof text !== "string"
    ) {

        return "";

    }


    let cleaned =
        text
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
        cleaned.indexOf(
            "{"
        );


    const lastBrace =
        cleaned.lastIndexOf(
            "}"
        );


    if (
        firstBrace >= 0 &&
        lastBrace >
        firstBrace
    ) {

        cleaned =
            cleaned.slice(
                firstBrace,
                lastBrace + 1
            );

    }


    return cleaned;

}


/*
 * =========================================================
 * NORMALIZE PLAN
 * =========================================================
 */


function normalizePlan(
    rawPlan
) {

    if (
        !rawPlan ||
        typeof rawPlan !== "object" ||
        Array.isArray(
            rawPlan
        )
    ) {

        return null;

    }


    const plan = {

        intent:
            typeof rawPlan.intent === "string"
                ? rawPlan.intent.trim()
                : "",

        requiresTools:
            rawPlan.requiresTools,

        reasoningSummary:
            typeof rawPlan.reasoningSummary === "string"
                ? rawPlan.reasoningSummary.trim()
                : "",

        steps:
            Array.isArray(
                rawPlan.steps
            )
                ? rawPlan.steps
                : []

    };


    if (
        plan.requiresTools === false
    ) {

        plan.steps =
            [];

    }


    /*
     * Нормализуем шаги.
     */
    plan.steps =
        plan.steps.map(
            (
                step,
                index
            ) => {

                if (
                    !step ||
                    typeof step !== "object" ||
                    Array.isArray(
                        step
                    )
                ) {

                    return step;

                }


                return {

                    id:
                        typeof step.id === "string" &&
                        step.id.trim()
                            ? step.id.trim()
                            : `step_${index + 1}`,

                    tool:
                        typeof step.tool === "string"
                            ? step.tool.trim()
                            : "",

                    arguments:
                        step.arguments &&
                        typeof step.arguments === "object" &&
                        !Array.isArray(
                            step.arguments
                        )
                            ? step.arguments
                            : {}

                };

            }
        );


    return plan;

}


/*
 * =========================================================
 * REFERENCE VALIDATION
 * =========================================================
 *
 * Проверяет все конструкции:
 *
 * {
 *     "$from": "search",
 *     "path": "data.results.0.url"
 * }
 *
 * Ссылка может вести ТОЛЬКО
 * на уже предыдущий шаг.
 */


function validateReferences(
    value,
    availableStepIds
) {

    if (
        value === null ||
        value === undefined ||
        typeof value !== "object"
    ) {

        return {
            success: true
        };

    }


    /*
     * -----------------------------------------------------
     * REFERENCE
     * -----------------------------------------------------
     */


    if (
        !Array.isArray(
            value
        ) &&
        Object.prototype.hasOwnProperty.call(
            value,
            "$from"
        )
    ) {

        if (
            typeof value.$from !== "string" ||
            !value.$from.trim()
        ) {

            return {
                success: false,

                text:
                    "Ссылка на предыдущий шаг содержит некорректный $from"
            };

        }


        const from =
            value.$from.trim();


        if (
            !availableStepIds.has(
                from
            )
        ) {

            return {
                success: false,

                text:
                    `Ссылка $from=${from} ведёт на неизвестный или будущий шаг`
            };

        }


        if (
            value.path !== undefined &&
            (
                typeof value.path !== "string" ||
                !value.path.trim()
            )
        ) {

            return {
                success: false,

                text:
                    `Некорректный path для ссылки на шаг ${from}`
            };

        }


        return {
            success: true
        };

    }


    /*
     * -----------------------------------------------------
     * ARRAY
     * -----------------------------------------------------
     */


    if (
        Array.isArray(
            value
        )
    ) {

        for (
            const item
            of value
        ) {

            const validation =
                validateReferences(
                    item,
                    availableStepIds
                );


            if (
                !validation.success
            ) {

                return validation;

            }

        }


        return {
            success: true
        };

    }


    /*
     * -----------------------------------------------------
     * OBJECT
     * -----------------------------------------------------
     */


    for (
        const item
        of Object.values(
            value
        )
    ) {

        const validation =
            validateReferences(
                item,
                availableStepIds
            );


        if (
            !validation.success
        ) {

            return validation;

        }

    }


    return {
        success: true
    };

}


/*
 * =========================================================
 * PLAN VALIDATION
 * =========================================================
 */


function validatePlan(
    plan
) {

    if (
        !plan ||
        typeof plan !== "object"
    ) {

        return {
            success: false,

            text:
                "План должен быть объектом"
        };

    }


    if (
        typeof plan.intent !== "string" ||
        !plan.intent.trim()
    ) {

        return {
            success: false,

            text:
                "В плане отсутствует intent"
        };

    }


    if (
        typeof plan.requiresTools !==
        "boolean"
    ) {

        return {
            success: false,

            text:
                "В плане отсутствует requiresTools"
        };

    }


    if (
        !Array.isArray(
            plan.steps
        )
    ) {

        return {
            success: false,

            text:
                "В плане отсутствует steps"
        };

    }


    /*
     * =====================================================
     * NO TOOLS
     * =====================================================
     */


    if (
        plan.requiresTools === false
    ) {

        if (
            plan.steps.length !== 0
        ) {

            return {
                success: false,

                text:
                    "При requiresTools=false список steps должен быть пустым"
            };

        }


        return {
            success: true
        };

    }


    /*
     * =====================================================
     * TOOLS REQUIRED
     * =====================================================
     */


    if (
        plan.steps.length === 0
    ) {

        return {
            success: false,

            text:
                "Planner указал requiresTools=true, но не создал шаги"
        };

    }


    if (
        plan.steps.length >
        MAX_STEPS
    ) {

        return {
            success: false,

            text:
                `Planner создал слишком много шагов: ${plan.steps.length}`
        };

    }


    const registeredTools =
        listTools();


    const toolNames =
        new Set(
            registeredTools.map(
                tool =>
                    tool.name
            )
        );


    const allStepIds =
        new Set();


    /*
     * Сначала проверяем уникальность ID.
     */


    for (
        let index = 0;
        index < plan.steps.length;
        index++
    ) {

        const step =
            plan.steps[index];


        if (
            !step ||
            typeof step !== "object" ||
            Array.isArray(
                step
            )
        ) {

            return {
                success: false,

                text:
                    `Некорректный шаг ${index + 1}`
            };

        }


        if (
            typeof step.id !== "string" ||
            !step.id.trim()
        ) {

            return {
                success: false,

                text:
                    `В шаге ${index + 1} отсутствует id`
            };

        }


        if (
            allStepIds.has(
                step.id
            )
        ) {

            return {
                success: false,

                text:
                    `Повторяющийся id шага: ${step.id}`
            };

        }


        allStepIds.add(
            step.id
        );

    }


    /*
     * Теперь проверяем каждый шаг
     * и зависимости по порядку.
     */


    const previousStepIds =
        new Set();


    for (
        let index = 0;
        index < plan.steps.length;
        index++
    ) {

        const step =
            plan.steps[index];


        if (
            typeof step.tool !== "string" ||
            !step.tool.trim()
        ) {

            return {
                success: false,

                text:
                    `В шаге ${index + 1} отсутствует tool`
            };

        }


        if (
            !toolNames.has(
                step.tool
            )
        ) {

            return {
                success: false,

                text:
                    `Planner выбрал неизвестный инструмент: ${step.tool}`
            };

        }


        if (
            !step.arguments ||
            typeof step.arguments !== "object" ||
            Array.isArray(
                step.arguments
            )
        ) {

            return {
                success: false,

                text:
                    `Некорректные arguments в шаге ${index + 1}`
            };

        }


        /*
         * Проверяем ссылки.
         *
         * Здесь доступны только уже выполненные
         * предыдущие шаги.
         */


        const referenceValidation =
            validateReferences(
                step.arguments,
                previousStepIds
            );


        if (
            !referenceValidation.success
        ) {

            return {
                success: false,

                text:
                    (
                        `Ошибка зависимостей в шаге ${step.id}: ` +
                        `${referenceValidation.text}`
                    )
            };

        }


        previousStepIds.add(
            step.id
        );

    }


    return {
        success: true
    };

}


/*
 * =========================================================
 * SHOULD RETRY AI ERROR
 * =========================================================
 */


function shouldRetryError(
    error
) {

    const status =
        Number(
            error?.status || 0
        );


    if (
        status === 429
    ) {

        return true;

    }


    if (
        status >= 500 &&
        status <= 599
    ) {

        return true;

    }


    const message =
        String(
            error?.message || ""
        ).toLowerCase();


    return (
        message.includes(
            "timeout"
        ) ||
        message.includes(
            "timed out"
        ) ||
        message.includes(
            "connection"
        ) ||
        message.includes(
            "network"
        )
    );

}


/*
 * =========================================================
 * ONE PLANNER ATTEMPT
 * =========================================================
 */


async function requestPlan(
    task,
    toolsDescription,
    attempt
) {

    const response =
        await groq.responses.create({

            model:
                "openai/gpt-oss-20b",

            instructions:
                (
                    "Ты — Planner системы Jessica Core. " +

                    "Ты НЕ отвечаешь пользователю и НЕ решаешь задачу. " +

                    "Твоя задача — построить исполняемый план. " +

                    "Сначала определи, нужны ли вообще инструменты. " +

                    "Если задача может быть выполнена AI самостоятельно " +
                    "без получения внешних актуальных данных и без внешнего действия, " +
                    "верни requiresTools=false и steps=[]. " +

                    "Если нужны инструменты, используй ТОЛЬКО инструменты " +
                    "из раздела ДОСТУПНЫЕ ИНСТРУМЕНТЫ. " +

                    "Никогда не придумывай инструменты. " +

                    "Каждый инструментальный шаг должен иметь уникальный id. " +

                    "Если следующему шагу нужен результат предыдущего, " +
                    "НЕ пытайся заранее придумать значение. " +

                    "Вместо этого используй ссылку на результат предыдущего шага. " +

                    "Формат ссылки: " +

                    JSON.stringify({
                        $from:
                            "id_предыдущего_шага",
                        path:
                            "путь.к.значению"
                    }) +

                    ". " +

                    "Например поиск страницы и последующее чтение страницы: " +

                    JSON.stringify({
                        intent:
                            "найти и прочитать источник",
                        requiresTools:
                            true,
                        reasoningSummary:
                            "сначала найти источник, затем открыть найденную страницу",
                        steps: [
                            {
                                id:
                                    "search",
                                tool:
                                    "web_search",
                                arguments: {
                                    query:
                                        "официальный сайт NASA актуальные миссии"
                                }
                            },
                            {
                                id:
                                    "fetch",
                                tool:
                                    "web_fetch",
                                arguments: {
                                    url: {
                                        $from:
                                            "search",
                                        path:
                                            "data.results.0.url"
                                    }
                                }
                            }
                        ]
                    }) +

                    ". " +

                    "Ссылка $from может вести ТОЛЬКО на предыдущий шаг. " +

                    "Нельзя ссылаться на будущий шаг. " +

                    "Нельзя использовать URL-заглушки вроде example.com, " +
                    "если реальный URL должен быть получен предыдущим инструментом. " +

                    "Если для ответа достаточно результатов web_search, " +
                    "не добавляй web_fetch без необходимости. " +

                    "Если пользователь прямо просит проверить информацию " +
                    "по самому сайту, официальному источнику, документу или странице, " +
                    "и URL заранее неизвестен, разумный маршрут может быть: " +
                    "web_search → web_fetch. " +

                    "Не используй специальные правила для конкретных сайтов, городов, " +
                    "тем или формулировок. Выбирай маршрут по смыслу задачи. " +

                    "Аргументы инструментов сохраняй максимально близко " +
                    "к смыслу запроса пользователя. " +

                    "reasoningSummary должен быть коротким описанием маршрута, " +
                    "а не скрытыми рассуждениями. " +

                    "Верни ТОЛЬКО один валидный JSON-объект. " +
                    "Без markdown и дополнительного текста."
                ),

            input:
                (
                    `ПОПЫТКА PLANNER: ${attempt}\n\n` +

                    `ДОСТУПНЫЕ ИНСТРУМЕНТЫ:\n` +
                    `${toolsDescription}\n\n` +

                    `ЗАДАЧА ПОЛЬЗОВАТЕЛЯ:\n` +
                    `${task}`
                ),

            reasoning: {
                effort:
                    "low"
            }

        });


    const raw =
        response.output_text
            ?.trim();


    if (!raw) {

        return {
            success: false,

            retryable:
                true,

            text:
                "Planner вернул
