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
 * 3. выбирает только реально зарегистрированные tools;
 * 4. строит план.
 *
 * Planner НЕ решает задачу пользователя.
 *
 *
 * ВАЖНО:
 *
 * Инструменты нужны не каждой задаче.
 *
 * Если задача может быть решена самим AI
 * без получения внешних данных
 * и без выполнения внешнего действия:
 *
 * requiresTools = false
 * steps = []
 *
 * После этого задача будет решена
 * Answer Composer.
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
            "Ты можешь создать только план без инструментов."
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


    /*
     * Если модель случайно добавила текст
     * до или после JSON, пытаемся извлечь
     * первый JSON-объект.
     */
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


    /*
     * Нормальный сценарий без инструментов.
     *
     * Даже если модель случайно не вернула
     * поле steps, оно становится [].
     */
    if (
        plan.requiresTools === false
    ) {

        plan.steps =
            [];

    }


    return plan;

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


    const registeredTools =
        listTools();


    const toolNames =
        new Set(
            registeredTools.map(
                tool =>
                    tool.name
            )
        );


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


        const toolName =
            typeof step.tool === "string"
                ? step.tool.trim()
                : "";


        if (!toolName) {

            return {
                success: false,

                text:
                    `В шаге ${index + 1} отсутствует tool`
            };

        }


        /*
         * Planner не может придумать
         * несуществующий инструмент.
         */
        if (
            !toolNames.has(
                toolName
            )
        ) {

            return {
                success: false,

                text:
                    `Planner выбрал неизвестный инструмент: ${toolName}`
            };

        }


        step.tool =
            toolName;


        if (
            step.arguments === undefined
        ) {

            step.arguments =
                {};

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


    /*
     * Rate limit.
     */
    if (
        status === 429
    ) {

        return true;

    }


    /*
     * Временные ошибки AI-сервиса.
     */
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

                    "Ты НЕ отвечаешь пользователю и НЕ решаешь его задачу. " +

                    "Тебе нужно только определить способ выполнения задачи. " +

                    "Сначала реши, нужны ли вообще внешние инструменты. " +

                    "Инструмент нужен только тогда, когда для выполнения задачи " +
                    "необходимо получить внешние или актуальные данные, " +
                    "прочитать внешний ресурс либо выполнить действие через доступный инструмент. " +

                    "Если задача может быть выполнена интеллектуально самой AI-моделью " +
                    "на основании текста пользователя и общих знаний, " +
                    "НЕ выбирай инструмент только ради того, чтобы выбрать инструмент. " +

                    "В таком случае обязательно установи requiresTools=false " +
                    "и верни steps=[]. " +

                    "Это нормальный и полноценный план. " +

                    "Если для решения действительно нужны инструменты, " +
                    "установи requiresTools=true и выбери только инструменты " +
                    "из списка ДОСТУПНЫЕ ИНСТРУМЕНТЫ. " +

                    "Никогда не придумывай инструмент, которого нет в списке. " +

                    "Если подходящего инструмента нет, " +
                    "не подменяй его другим неподходящим инструментом. " +

                    "Если задача требует актуальной информации " +
                    "и имеется подходящий инструмент получения такой информации, " +
                    "используй его вместо памяти модели. " +

                    "Если требуется несколько инструментальных действий, " +
                    "создай несколько последовательных шагов. " +

                    "Не разбивай обычную интеллектуальную работу AI " +
                    "на фиктивные tool-шаги. " +

                    "Аргументы инструментов сохраняй максимально близко " +
                    "к исходной формулировке пользователя. " +

                    "Например географическое название передавай как географическое название, " +
                    "не преобразовывай его самостоятельно в технические идентификаторы, " +
                    "если это должен сделать сам инструмент. " +

                    "reasoningSummary — только короткое описание выбранного маршрута. " +
                    "Не раскрывай скрытые рассуждения. " +

                    "Ответ должен содержать ТОЛЬКО один валидный JSON-объект. " +
                    "Без markdown, комментариев и текста вокруг JSON. " +

                    "Формат для задачи БЕЗ инструментов: " +

                    JSON.stringify({
                        intent:
                            "краткий идентификатор намерения",
                        requiresTools:
                            false,
                        reasoningSummary:
                            "задача может быть выполнена AI без внешних инструментов",
                        steps:
                            []
                    }) +

                    " Формат для задачи С инструментами: " +

                    JSON.stringify({
                        intent:
                            "краткий идентификатор намерения",
                        requiresTools:
                            true,
                        reasoningSummary:
                            "краткое описание маршрута",
                        steps: [
                            {
                                tool:
                                    "точное имя зарегистрированного инструмента",
                                arguments:
                                    {}
                            }
                        ]
                    })
                ),

            input:
                (
                    `ПОПЫТКА PLANNER: ${attempt}\n\n` +

                    `ДОСТУПНЫЕ ИНСТРУМЕНТЫ:\n` +
                    `${toolsDescription}\n\n` +

                    `ЗАДАЧА ПОЛЬЗОВАТЕЛЯ:\n` +
                    `${task}`
                ),

            /*
             * Planner не должен долго рассуждать.
             * Его задача — маршрутизация.
             */
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
                "Planner вернул пустой ответ"
        };

    }


    let parsed;


    try {

        parsed =
            JSON.parse(
                cleanJsonText(
                    raw
                )
            );

    } catch (error) {

        console.error(
            "Planner invalid JSON:",
            raw
        );


        return {
            success: false,

            retryable:
                true,

            text:
                "Planner вернул некорректный JSON"
        };

    }


    const plan =
        normalizePlan(
            parsed
        );


    const validation =
        validatePlan(
            plan
        );


    if (
        !validation.success
    ) {

        console.error(
            "Planner validation error:",
            validation.text,
            plan
        );


        /*
         * Даём модели возможность
         * исправить собственный план
         * следующей попыткой.
         */
        return {
            success: false,

            retryable:
                true,

            text:
                validation.text
        };

    }


    return {
        success: true,

        plan
    };

}


/*
 * =========================================================
 * CREATE PLAN
 * =========================================================
 */


export async function createPlan(
    task
) {

    const normalizedTask =
        typeof task === "string"
            ? task.trim()
            : "";


    if (!normalizedTask) {

        return {
            success: false,

            text:
                "Planner получил пустую задачу"
        };

    }


    if (!groq) {

        return {
            success: false,

            text:
                "Groq Planner не настроен"
        };

    }


    const toolsDescription =
        buildToolsDescription();


    let lastErrorText =
        "Planner не смог проанализировать задачу";


    for (
        let attempt = 1;
        attempt <= MAX_ATTEMPTS;
        attempt++
    ) {

        try {

            console.log(
                `Jessica Planner attempt ${attempt}/${MAX_ATTEMPTS}`
            );


            const result =
                await requestPlan(
                    normalizedTask,
                    toolsDescription,
                    attempt
                );


            if (
                result.success
            ) {

                console.log(
                    "Jessica Planner plan:",
                    JSON.stringify(
                        result.plan
                    )
                );


                return result;

            }


            lastErrorText =
                result.text ||
                lastErrorText;


            if (
                result.retryable !== true ||
                attempt === MAX_ATTEMPTS
            ) {

                break;

            }


        } catch (error) {

            console.error(
                `Planner attempt ${attempt} error:`,
                error
            );


            lastErrorText =
                error?.message
                    ? `Planner AI error: ${error.message}`
                    : "Planner не смог проанализировать задачу";


            if (
                attempt === MAX_ATTEMPTS ||
                !shouldRetryError(
                    error
                )
            ) {

                break;

            }

        }


        /*
         * Небольшая пауза перед повтором,
         * чтобы не ударяться сразу
         * в тот же rate limit / временный сбой.
         */
        await sleep(
            RETRY_DELAY_MS *
            attempt
        );

    }


    return {
        success: false,

        text:
            lastErrorText
    };

}
