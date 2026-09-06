import OpenAI from "openai";

import {
    listTools
} from "../tools/toolRegistry.js";

import {
    buildPlannerInstructions
} from "./planner/plannerPrompt.js";

import {
    normalizePlan
} from "./planner/planNormalizer.js";

import {
    validatePlan
} from "./planner/planValidator.js";

import {
    MAX_PLANNER_ATTEMPTS,
    sleep,
    isRetryablePlannerError,
    getPlannerRetryDelay
} from "./planner/plannerRetry.js";


/*
 * =========================================================
 * JESSICA PLANNER
 * =========================================================
 *
 * Главный координатор планирования.
 *
 * Детальная логика вынесена в:
 *
 * core/planner/
 *
 * Этот файл должен оставаться небольшим.
 */


const groq =
    new OpenAI({
        apiKey:
            process.env.GROQ_API_KEY,

        baseURL:
            "https://api.groq.com/openai/v1"
    });


const PLANNER_MODEL =
    "openai/gpt-oss-20b";


/*
 * =========================================================
 * CLEAN JSON
 * =========================================================
 */


function cleanJsonText(text) {

    let value =
        String(text || "").trim();


    if (
        value.startsWith("```")
    ) {

        value =
            value.replace(
                /^```(?:json)?\s*/i,
                ""
            );

        value =
            value.replace(
                /\s*```$/,
                ""
            );

    }


    const firstBrace =
        value.indexOf("{");

    const lastBrace =
        value.lastIndexOf("}");


    if (
        firstBrace !== -1 &&
        lastBrace > firstBrace
    ) {

        value =
            value.slice(
                firstBrace,
                lastBrace + 1
            );

    }


    return value.trim();

}


/*
 * =========================================================
 * TOOL DESCRIPTION
 * =========================================================
 */


function buildToolsText() {

    const tools =
        listTools();


    if (
        tools.length === 0
    ) {

        return "Инструменты отсутствуют.";

    }


    return tools
        .map(
            tool => {

                return JSON.stringify({
                    name:
                        tool.name,

                    description:
                        tool.description,

                    arguments:
                        tool.arguments || {}
                });

            }
        )
        .join("\n");

}


/*
 * =========================================================
 * AI REQUEST
 * =========================================================
 */


async function requestPlan(
    task,
    previousError = ""
) {

    const instructions =
        buildPlannerInstructions();


    const toolsText =
        buildToolsText();


    const retryContext =
        previousError
            ? [
                "",
                "ПРЕДЫДУЩИЙ ПЛАН БЫЛ ОТКЛОНЁН:",
                previousError,
                "",
                "Исправь ошибку и создай новый валидный план."
            ].join("\n")
            : "";


    const response =
        await groq.chat.completions.create({
            model:
                PLANNER_MODEL,

            temperature:
                0,

            messages: [
                {
                    role:
                        "system",

                    content:
                        instructions
                },
                {
                    role:
                        "user",

                    content: [
                        "ЗАДАЧА:",
                        String(task || "").trim(),

                        "",
                        "ДОСТУПНЫЕ ИНСТРУМЕНТЫ:",
                        toolsText,

                        retryContext
                    ].join("\n")
                }
            ]
        });


    return (
        response
            ?.choices
            ?.[0]
            ?.message
            ?.content || ""
    );

}


/*
 * =========================================================
 * PARSE PLAN
 * =========================================================
 */


function parsePlan(
    text
) {

    const cleaned =
        cleanJsonText(
            text
        );


    if (!cleaned) {

        throw new Error(
            "Planner вернул пустой ответ"
        );

    }


    try {

        return JSON.parse(
            cleaned
        );

    } catch {

        throw new Error(
            "Planner вернул невалидный JSON"
        );

    }

}


/*
 * =========================================================
 * CREATE PLAN
 * =========================================================
 */


export async function createPlan(
    task
) {

    const cleanTask =
        String(task || "").trim();


    if (!cleanTask) {

        return {
            success:
                false,

            text:
                "Задача для Planner не указана"
        };

    }


    if (
        !process.env.GROQ_API_KEY
    ) {

        return {
            success:
                false,

            text:
                "GROQ_API_KEY не настроен"
        };

    }


    let lastError =
        "Не удалось построить план";


    for (
        let attempt = 1;
        attempt <= MAX_PLANNER_ATTEMPTS;
        attempt++
    ) {

        try {

            const rawText =
                await requestPlan(
                    cleanTask,
                    attempt > 1
                        ? lastError
                        : ""
                );


            const rawPlan =
                parsePlan(
                    rawText
                );


            const plan =
                normalizePlan(
                    rawPlan
                );


            if (!plan) {

                throw new Error(
                    "Не удалось нормализовать план"
                );

            }


            const validation =
                validatePlan(
                    plan
                );


            if (
                !validation.success
            ) {

                lastError =
                    validation.text ||
                    "План не прошёл проверку";


                console.warn(
                    `Planner validation failed ` +
                    `[${attempt}/${MAX_PLANNER_ATTEMPTS}]:`,
                    lastError
                );


                continue;

            }


            console.log(
                "Jessica plan:",
                JSON.stringify(
                    plan
                )
            );


            return {
                success:
                    true,

                plan
            };

        } catch (error) {

            lastError =
                error?.message ||
                "Неизвестная ошибка Planner";


            console.error(
                `Planner error ` +
                `[${attempt}/${MAX_PLANNER_ATTEMPTS}]:`,
                lastError
            );


            if (
                attempt <
                    MAX_PLANNER_ATTEMPTS &&
                isRetryablePlannerError(
                    error
                )
            ) {

                await sleep(
                    getPlannerRetryDelay(
                        attempt
                    )
                );

            }

        }

    }


    return {
        success:
            false,

        text:
            `Planner не смог создать корректный план: ${lastError}`
    };

}


/*
 * =========================================================
 * BACKWARD-COMPATIBLE EXPORT
 * =========================================================
 */


export async function planTask(
    task
) {

    const result =
        await createPlan(
            task
        );


    if (
        result.success
    ) {

        return result.plan;

    }


    throw new Error(
        result.text
    );

}
