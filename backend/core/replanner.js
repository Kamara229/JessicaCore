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


/*
 * =========================================================
 * JESSICA REPLANNER
 * =========================================================
 *
 * Строит альтернативный план после того,
 * как Validator отклонил предыдущий результат.
 *
 * Получает:
 *
 * - исходную задачу;
 * - предыдущий план;
 * - причину отказа Validator;
 * - уже выполненные результаты.
 *
 * Не выполняет инструменты сам.
 */


const groq =
    process.env.GROQ_API_KEY
        ? new OpenAI({
            apiKey: process.env.GROQ_API_KEY,
            baseURL: "https://api.groq.com/openai/v1"
        })
        : null;


const MODEL =
    "openai/gpt-oss-20b";


/*
 * =========================================================
 * JSON CLEANUP
 * =========================================================
 */


function cleanJsonText(text) {

    let value =
        String(text || "")
            .replace(/```json/gi, "")
            .replace(/```/g, "")
            .trim();


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


    return value;
}


/*
 * =========================================================
 * TOOL DESCRIPTION
 * =========================================================
 */


function buildToolsText() {

    const tools =
        listTools();


    return tools
        .map(
            tool =>
                JSON.stringify({
                    name:
                        tool.name,

                    description:
                        tool.description,

                    arguments:
                        tool.arguments || {}
                })
        )
        .join("\n");
}


/*
 * =========================================================
 * CREATE ALTERNATIVE PLAN
 * =========================================================
 */


export async function replanTask(
    task,
    previousPlan,
    validationResult,
    previousRunResult
) {

    if (!groq) {

        return {
            success: false,
            reason:
                "Replanner недоступен: GROQ_API_KEY не настроен"
        };
    }


    const validatorReason =
        String(
            validationResult?.reason || ""
        ).trim();


    try {

        const response =
            await groq.chat.completions.create({

                model:
                    MODEL,

                temperature:
                    0,

                messages: [
                    {
                        role:
                            "system",

                        content: [
                            buildPlannerInstructions(),

                            "",
                            "ДОПОЛНИТЕЛЬНАЯ РОЛЬ:",
                            "Предыдущий план уже был выполнен и отклонён Validator.",
                            "Создай АЛЬТЕРНАТИВНЫЙ план.",
                            "",
                            "Не повторяй тот же неудачный маршрут без необходимости.",
                            "Используй причину отказа Validator как обратную связь.",
                            "",
                            "Если предыдущий источник не содержал нужного факта,",
                            "измени поиск, выбери более конкретный источник",
                            "или построй дополнительные шаги.",
                            "",
                            "Если конкретное утверждение не подтверждено,",
                            "получи более подходящее evidence.",
                            "",
                            "Не придумывай результат заранее.",
                            "Верни только валидный JSON."
                        ].join("\n")
                    },

                    {
                        role:
                            "user",

                        content: [
                            "ИСХОДНАЯ ЗАДАЧА:",
                            String(task || ""),

                            "",
                            "ПРЕДЫДУЩИЙ ПЛАН:",
                            JSON.stringify(
                                previousPlan,
                                null,
                                2
                            ),

                            "",
                            "ПРИЧИНА ОТКЛОНЕНИЯ VALIDATOR:",
                            validatorReason,

                            "",
                            "ПРЕДЫДУЩИЕ РЕЗУЛЬТАТЫ ВЫПОЛНЕНИЯ:",
                            JSON.stringify(
                                previousRunResult,
                                null,
                                2
                            ),

                            "",
                            "ДОСТУПНЫЕ ИНСТРУМЕНТЫ:",
                            buildToolsText()
                        ].join("\n")
                    }
                ]
            });


        const raw =
            response
                ?.choices
                ?.[0]
                ?.message
                ?.content;


        if (!raw) {

            return {
                success: false,
                reason:
                    "Replanner вернул пустой ответ"
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
                "Replanner invalid JSON:",
                raw
            );


            return {
                success: false,
                reason:
                    "Replanner вернул некорректный JSON"
            };
        }


        const plan =
            normalizePlan(
                parsed
            );


        if (!plan) {

            return {
                success: false,
                reason:
                    "Не удалось нормализовать альтернативный план"
            };
        }


        const validation =
            validatePlan(
                plan
            );


        if (
            validation.success !== true
        ) {

            return {
                success: false,
                reason:
                    validation.text ||
                    "Альтернативный план не прошёл проверку"
            };
        }


        console.log(
            "Jessica Replan:",
            JSON.stringify(plan)
        );


        return {
            success: true,
            plan
        };


    } catch (error) {

        console.error(
            "Replanner error:",
            error
        );


        return {
            success: false,
            reason:
                error?.message ||
                "Ошибка Replanner"
        };
    }

}
