import OpenAI from "openai";

import {
    listTools
} from "../tools/toolRegistry.js";


/*
 * =========================================================
 * JESSICA TASK PLANNER
 * =========================================================
 *
 * Planner НЕ решает задачу пользователя.
 *
 * Его задача:
 *
 * 1. понять намерение;
 * 2. определить, нужны ли инструменты;
 * 3. выбрать инструменты;
 * 4. сформировать последовательность шагов.
 *
 * Список инструментов берётся из Tool Registry.
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
 * TOOLS DESCRIPTION
 * =========================================================
 */


function buildToolsDescription() {

    const tools =
        listTools();


    if (
        tools.length === 0
    ) {

        return "Инструменты пока не зарегистрированы.";

    }


    return tools
        .map(
            tool =>
                JSON.stringify(
                    tool,
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

    return text
        .replace(
            /```json/gi,
            ""
        )
        .replace(
            /```/g,
            ""
        )
        .trim();

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
        typeof plan.intent !==
        "string" ||
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
     * Если инструменты не нужны,
     * шагов быть не должно.
     */
    if (
        !plan.requiresTools &&
        plan.steps.length > 0
    ) {

        return {
            success: false,
            text:
                "Planner указал steps при requiresTools=false"
        };

    }


    /*
     * Если инструменты нужны,
     * должен быть хотя бы один шаг.
     */
    if (
        plan.requiresTools &&
        plan.steps.length === 0
    ) {

        return {
            success: false,
            text:
                "Planner не указал шаги для инструментальной задачи"
        };

    }


    const tools =
        listTools();


    const toolNames =
        new Set(
            tools.map(
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
            typeof step !== "object"
        ) {

            return {
                success: false,
                text:
                    `Некорректный шаг ${index + 1}`
            };

        }


        if (
            typeof step.tool !==
            "string" ||
            !step.tool.trim()
        ) {

            return {
                success: false,
                text:
                    `В шаге ${index + 1} отсутствует tool`
            };

        }


        /*
         * Planner не имеет права придумать
         * инструмент, которого нет в Registry.
         */
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
            step.arguments === undefined
        ) {

            step.arguments = {};

        }


        if (
            typeof step.arguments !== "object" ||
            Array.isArray(
                step.arguments
            ) ||
            step.arguments === null
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
 * CREATE PLAN
 * =========================================================
 */


export async function createPlan(
    task
) {

    if (
        typeof task !== "string" ||
        !task.trim()
    ) {

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


    try {

        const toolsDescription =
            buildToolsDescription();


        const response =
            await groq.responses.create({

                model:
                    "openai/gpt-oss-20b",

                instructions:
                    (
                        "Ты — Planner системы Jessica Core. " +

                        "Ты НЕ отвечаешь пользователю. " +
                        "Ты НЕ должен решать задачу самостоятельно. " +

                        "Твоя работа — понять намерение пользователя " +
                        "и создать план выполнения задачи. " +

                        "Ты можешь выбирать ТОЛЬКО инструменты, " +
                        "которые перечислены в разделе ДОСТУПНЫЕ ИНСТРУМЕНТЫ. " +

                        "Никогда не придумывай название нового инструмента. " +

                        "Если задача не требует внешних или специализированных инструментов, " +
                        "установи requiresTools=false и верни пустой steps. " +

                        "Если пользователь спрашивает актуальную информацию, " +
                        "не пытайся ответить на неё из памяти модели, " +
                        "если для неё имеется подходящий инструмент. " +

                        "Если задача требует нескольких действий, " +
                        "создай несколько последовательных шагов. " +

                        "Сохраняй значения аргументов максимально близко " +
                        "к формулировке пользователя. " +

                        "Например, если пользователь написал «в Дубае», " +
                        "не пытайся самостоятельно преобразовывать это в timezone. " +
                        "Передай location инструменту как географическое значение. " +

                        "reasoningSummary должен содержать только краткое описание маршрута, " +
                        "а не скрытые рассуждения. " +

                        "Возвращай ТОЛЬКО валидный JSON. " +
                        "Никакого markdown и никакого дополнительного текста. " +

                        "Формат ответа: " +

                        JSON.stringify({

                            intent:
                                "краткий идентификатор намерения",

                            requiresTools:
                                true,

                            reasoningSummary:
                                "краткое описание выбранного маршрута",

                            steps: [

                                {
                                    tool:
                                        "имя зарегистрированного инструмента",

                                    arguments: {}
                                }

                            ]

                        })
                    ),

                input:
                    (
                        `ДОСТУПНЫЕ ИНСТРУМЕНТЫ:\n` +
                        `${toolsDescription}\n\n` +

                        `ЗАДАЧА ПОЛЬЗОВАТЕЛЯ:\n` +
                        `${task}`
                    ),

                reasoning: {
                    effort:
                        "medium"
                }

            });


        const raw =
            response.output_text
                ?.trim();


        if (!raw) {

            return {
                success: false,
                text:
                    "Planner вернул пустой ответ"
            };

        }


        let plan;


        try {

            plan =
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
                text:
                    "Planner вернул некорректный JSON"
            };

        }


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


            return {
                success: false,
                text:
                    validation.text
            };

        }


        return {
            success: true,
            plan
        };


    } catch (error) {

        console.error(
            "Planner error:",
            error
        );


        return {
            success: false,

            status:
                error?.status || 0,

            text:
                "Planner не смог проанализировать задачу"
        };

    }

}
