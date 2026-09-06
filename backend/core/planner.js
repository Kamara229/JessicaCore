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
 * 1. понимает цель;
 * 2. определяет, нужны ли инструменты;
 * 3. определяет требуемый уровень доказательств;
 * 4. создаёт исполняемый план;
 * 5. связывает шаги через $from.
 *
 * Planner НЕ отвечает пользователю.
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


/*
 * =========================================================
 * TOOLS
 * =========================================================
 */


function buildToolsDescription() {

    return JSON.stringify(
        listTools().map(
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
 * NORMALIZE EVIDENCE
 * =========================================================
 */


function normalizeEvidence(
    raw
) {

    const allowedModes =
        new Set([
            "none",
            "search_results",
            "source_content"
        ]);


    const mode =
        allowedModes.has(
            raw?.mode
        )
            ? raw.mode
            : "none";


    return {
        mode,

        reason:
            typeof raw?.reason === "string"
                ? raw.reason.trim()
                : ""
    };

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


    const steps =
        requiresTools === false
            ? []
            : (
                Array.isArray(
                    raw.steps
                )
                    ? raw.steps.map(
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
                    )
                    : []
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

        evidence:
            normalizeEvidence(
                raw.evidence
            ),

        steps
    };

}


/*
 * =========================================================
 * REFERENCES
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


    if (!plan.intent) {

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
        !plan.evidence ||
        typeof plan.evidence !== "object"
    ) {

        return {
            success: false,

            text:
                "В плане отсутствует evidence"
        };

    }


    const allowedEvidenceModes =
        new Set([
            "none",
            "search_results",
            "source_content"
        ]);


    if (
        !allowedEvidenceModes.has(
            plan.evidence.mode
        )
    ) {

        return {
            success: false,

            text:
                `Некорректный evidence.mode: ${plan.evidence.mode}`
        };

    }


    /*
     * Без инструментов.
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


        if (
            plan.evidence.mode !==
            "none"
        ) {

            return {
                success: false,

                text:
                    "Задача без инструментов не может требовать внешние доказательства"
            };

        }


        return {
            success: true
        };

    }


    /*
     * С инструментами.
     */
    if (
        plan.steps.length === 0
    ) {

        return {
            success: false,

            text:
                "Planner не создал шаги"
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


    const registeredTools =
        new Set(
            listTools().map(
                tool =>
                    tool.name
            )
        );


    const allIds =
        new Set();


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


    const previousIds =
        new Set();


    for (
        const step
        of plan.steps
    ) {

        if (
            !registeredTools.has(
                step.tool
            )
        ) {

            return {
                success: false,

                text:
                    `Неизвестный инструмент: ${step.tool}`
            };

        }


        const references =
            validateReferences(
                step.arguments,
                previousIds
            );


        if (
            !references.success
        ) {

            return {
                success: false,

                text:
                    `Ошибка зависимостей шага ${step.id}: ${references.text}`
            };

        }


        previousIds.add(
            step.id
        );

    }


    /*
     * =====================================================
     * EVIDENCE CONSISTENCY
     * =====================================================
     */


    const usedTools =
        new Set(
            plan.steps.map(
                step =>
                    step.tool
            )
        );


    /*
     * Если требуется содержимое источника,
     * plan обязан реально включать web_fetch.
     */
    if (
        plan.evidence.mode ===
        "source_content" &&
        !usedTools.has(
            "web_fetch"
        )
    ) {

        return {
            success: false,

            text:
                (
                    "План требует evidence=source_content, " +
                    "но не содержит web_fetch"
                )
        };

    }


    /*
     * Если нужны результаты поиска,
     * должен быть web_search.
     */
    if (
        plan.evidence.mode ===
        "search_results" &&
        !usedTools.has(
            "web_search"
        )
    ) {

        return {
            success: false,

            text:
                (
                    "План требует evidence=search_results, " +
                    "но не содержит web_search"
                )
        };

    }


    return {
        success: true
    };

}


/*
 * =========================================================
 * RETRY
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

    const response =
        await groq.responses.create({

            model:
                "openai/gpt-oss-20b",

            instructions:
                [
                    "Ты Planner системы Jessica Core.",
                    "Ты не отвечаешь пользователю.",
                    "Ты создаёшь только исполняемый план.",
                    "",
                    "Определи requiresTools.",
                    "",
                    "Также обязательно определи evidence.mode:",
                    "",
                    "none — внешние доказательства не нужны.",
                    "",
                    "search_results — достаточно результатов интернет-поиска.",
                    "",
                    "source_content — обязательно нужно прочитать содержимое конкретного источника, страницы, сайта или документа.",
                    "",
                    "Если пользователь просит узнать что-то по самому сайту,",
                    "по документу, по странице, по официальному источнику",
                    "или проверить содержимое источника,",
                    "используй evidence.mode=source_content.",
                    "",
                    "Если достаточно поисковой выдачи,",
                    "используй evidence.mode=search_results.",
                    "",
                    "Если внешние данные вообще не нужны,",
                    "requiresTools=false, evidence.mode=none, steps=[].",
                    "",
                    "Если evidence.mode=source_content,",
                    "в плане обязательно должен присутствовать web_fetch.",
                    "",
                    "Если URL заранее неизвестен:",
                    "сначала web_search, затем web_fetch через $from.",
                    "",
                    "Формат зависимости:",
                    '{"$from":"search","path":"data.results.0.url"}',
                    "",
                    "Используй только зарегистрированные инструменты.",
                    "Не придумывай URL.",
                    "Не придумывай инструменты.",
                    "",
                    "Каждый шаг имеет уникальный id.",
                    "",
                    "Верни только JSON.",
                    "",
                    "Структура:",
                    JSON.stringify(
                        {
                            intent:
                                "intent_name",

                            requiresTools:
                                true,

                            reasoningSummary:
                                "краткий маршрут",

                            evidence: {
                                mode:
                                    "source_content",

                                reason:
                                    "почему нужен такой уровень подтверждения"
                            },

                            steps: [
                                {
                                    id:
                                        "search",

                                    tool:
                                        "web_search",

                                    arguments: {
                                        query:
                                            "поисковый запрос"
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
                        }
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

    
