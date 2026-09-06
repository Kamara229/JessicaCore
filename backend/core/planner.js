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
 * - понимает цель задачи;
 * - решает, нужны ли инструменты;
 * - выбирает только зарегистрированные инструменты;
 * - строит многошаговый план;
 * - умеет передавать результаты между шагами через:
 *
 * {
 *   "$from": "search",
 *   "path": "data.results.0.url"
 * }
 *
 * Planner сам задачу НЕ решает.
 */


/*
 * =========================================================
 * AI
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


const MAX_ATTEMPTS = 3;
const MAX_STEPS = 15;
const RETRY_DELAY_MS = 1200;


/*
 * =========================================================
 * HELPERS
 * =========================================================
 */


function sleep(
    ms
) {

    return new Promise(
        resolve =>
            setTimeout(
                resolve,
                ms
            )
    );

}


function buildToolsDescription() {

    const tools =
        listTools();


    if (
        tools.length === 0
    ) {

        return "Зарегистрированных инструментов нет.";

    }


    return JSON.stringify(
        tools.map(
            tool => ({
                name:
                    tool.name,

                description:
                    tool.description,

                arguments:
                    tool.arguments || {}
            })
        ),
        null,
        2
    );

}


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


    const start =
        cleaned.indexOf(
            "{"
        );


    const end =
        cleaned.lastIndexOf(
            "}"
        );


    if (
        start >= 0 &&
        end > start
    ) {

        cleaned =
            cleaned.slice(
                start,
                end + 1
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
    raw
) {

    if (
        !raw ||
        typeof raw !== "object" ||
        Array.isArray(
            raw
        )
    ) {

        return null;

    }


    const requiresTools =
        raw.requiresTools;


    const rawSteps =
        Array.isArray(
            raw.steps
        )
            ? raw.steps
            : [];


    const steps =
        requiresTools === false
            ? []
            : rawSteps.map(
                (
                    step,
                    index
                ) => ({
                    id:
                        typeof step?.id === "string" &&
                        step.id.trim()
                            ? step.id.trim()
                            : `step_${index + 1}`,

                    tool:
                        typeof step?.tool === "string"
                            ? step.tool.trim()
                            : "",

                    arguments:
                        step?.arguments &&
                        typeof step.arguments === "object" &&
                        !Array.isArray(
                            step.arguments
                        )
                            ? step.arguments
                            : {}
                })
            );


    return {
        intent:
            typeof raw.intent === "string"
                ? raw.intent.trim()
                : "",

        requiresTools,

        reasoningSummary:
            typeof raw.reasoningSummary === "string"
                ? raw.reasoningSummary.trim()
                : "",

        steps
    };

}


/*
 * =========================================================
 * REFERENCE VALIDATION
 * =========================================================
 */


function validateReferences(
    value,
    previousStepIds
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
     * Ссылка на предыдущий шаг.
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

        const from =
            typeof value.$from === "string"
                ? value.$from.trim()
                : "";


        if (!from) {

            return {
                success: false,

                text:
                    "Некорректный $from"
            };

        }


        if (
            !previousStepIds.has(
                from
            )
        ) {

            return {
                success: false,

                text:
                    `Ссылка ведёт на неизвестный или будущий шаг: ${from}`
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
                    `Некорректный path для шага ${from}`
            };

        }


        return {
            success: true
        };

    }


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
                    previousStepIds
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


    for (
        const item
        of Object.values(
            value
        )
    ) {

        const validation =
            validateReferences(
                item,
                previousStepIds
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

    if (!plan) {

        return {
            success: false,

            text:
                "Некорректный план"
        };

    }


    if (
        typeof plan.intent !== "string" ||
        !plan.intent
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
     * Задача без инструментов.
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
                    "При requiresTools=false steps должен быть пустым"
            };

        }


        return {
            success: true
        };

    }


    /*
     * Инструменты нужны.
     */
    if (
        plan.steps.length === 0
    ) {

        return {
            success: false,

            text:
                "Planner не создал инструментальные шаги"
        };

    }


    if (
        plan.steps.length >
        MAX_STEPS
    ) {

        return {
            success: false,

            text:
                `Слишком много шагов: ${plan.steps.length}`
        };

    }


    const registeredToolNames =
        new Set(
            listTools().map(
                tool =>
                    tool.name
            )
        );


    const allIds =
        new Set();


    /*
     * Сначала уникальность ID.
     */
    for (
        const step
        of plan.steps
    ) {

        if (
            !step.id
        ) {

            return {
                success: false,

                text:
                    "У шага отсутствует id"
            };

        }


        if (
            allIds.has(
                step.id
            )
        ) {

            return {
                success: false,

                text:
                    `Повторяется id шага: ${step.id}`
            };

        }


        allIds.add(
            step.id
        );

    }


    /*
     * Потом инструменты и зависимости.
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
            !registeredToolNames.has(
                step.tool
            )
        ) {

            return {
                success: false,

                text:
                    `Неизвестный инструмент: ${step.tool}`
            };

        }


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
                        `Ошибка зависимостей шага ${step.id}: ` +
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
 * RETRY CHECK
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
        status === 429 ||
        status >= 500
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
            "network"
        ) ||
        message.includes(
            "connection"
        )
    );

}


/*
 * =========================================================
 * REQUEST PLAN
 * =========================================================
 */


async function requestPlan(
    task,
    toolsDescription
) {

    const examplePlan = {
        intent:
            "find_and_read_source",

        requiresTools:
            true,

        reasoningSummary:
            "Сначала найти страницу, затем открыть найденный источник.",

        steps: [
            {
                id:
                    "search",

                tool:
                    "web_search",

                arguments: {
                    query:
                        "официальный источник с нужной информацией"
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
    };


    const noToolsExample = {
        intent:
            "intellectual_task",

        requiresTools:
            false,

        reasoningSummary:
            "Внешние инструменты не требуются.",

        steps:
            []
    };


    const response =
        await groq.responses.create({

            model:
                "openai/gpt-oss-20b",

            instructions:
                [
                    "Ты Planner системы Jessica Core.",
                    "Ты не отвечаешь пользователю и не решаешь задачу.",
                    "Ты создаёшь только исполняемый план.",
                    "",
                    "Если внешние данные или действия не нужны:",
                    "requiresTools=false и steps=[].",
                    "",
                    "Если нужны инструменты:",
                    "используй только инструменты из списка.",
                    "Не придумывай инструменты.",
                    "",
                    "Каждый инструментальный шаг имеет уникальный id.",
                    "",
                    "Если следующему шагу нужен результат предыдущего,",
                    "используй объект:",
                    '{"$from":"id_шага","path":"data.path"}',
                    "",
                    "Ссылка может вести только на предыдущий шаг.",
                    "",
                    "Если URL заранее неизвестен, не придумывай его.",
                    "Сначала получи URL поиском, затем передай его через $from.",
                    "",
                    "Если достаточно поисковых результатов, web_fetch не обязателен.",
                    "Если пользователь просит проверить сам сайт, страницу, документ",
                    "или официальный источник, может понадобиться search → fetch.",
                    "",
                    "Не используй правила под конкретные города, сайты или темы.",
                    "Выбирай маршрут по смыслу задачи.",
                    "",
                    "Верни только JSON без markdown.",
                    "",
                    "Пример плана без инструментов:",
                    JSON.stringify(
                        noToolsExample
                    ),
                    "",
                    "Пример зависимого плана:",
                    JSON.stringify(
                        examplePlan
                    )
                ].join(
                    "\n"
                ),

            input:
                (
                    `ДОСТУПНЫЕ ИНСТРУМЕНТЫ:\n` +
                    `${toolsDescription}\n\n` +
                    `ЗАДАЧА:\n` +
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

    } catch {

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
            "Planner validation failed:",
            validation.text,
            JSON.stringify(
                plan
            )
        );


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


    let lastError =
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
                    toolsDescription
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


            lastError =
                result.text ||
                lastError;


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


            lastError =
                error?.message ||
                lastError;


            if (
                attempt === MAX_ATTEMPTS ||
                !shouldRetryError(
                    error
                )
            ) {

                break;

            }

        }


        await sleep(
            RETRY_DELAY_MS *
            attempt
        );

    }


    return {
        success: false,

        text:
            lastError
    };

                    }
