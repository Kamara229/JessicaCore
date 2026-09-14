import {
    getGroqClient
} from "../../ai/groqClient.js";

import {
    buildPlannerInstructions
} from "./plannerPrompt.js";

import {
    buildToolsText
} from "./plannerTools.js";

import {
    buildPlanningContextText
} from "./planningContextText.js";


/*
 * =========================================================
 * JESSICA PLANNER REQUEST
 * =========================================================
 *
 * Отвечает только за запрос
 * к AI-модели Planner.
 *
 *
 * Здесь находятся:
 *
 * - выбор модели Planner;
 * - сбор prompt;
 * - передача задачи;
 * - передача PlanningContext;
 * - передача доступных инструментов;
 * - передача ошибки предыдущей попытки.
 *
 *
 * Groq Client находится отдельно:
 *
 * backend/ai/groqClient.js
 *
 *
 * Этот модуль НЕ содержит:
 *
 * - создание AI-клиента;
 * - parsing JSON;
 * - нормализацию плана;
 * - валидацию плана;
 * - retry-цикл;
 * - поиск Experience;
 * - сохранение Experience;
 * - выполнение инструментов.
 *
 * =========================================================
 */


/*
 * =========================================================
 * MODEL
 * =========================================================
 */


const PLANNER_MODEL =
    "openai/gpt-oss-20b";


/*
 * =========================================================
 * REQUEST PLAN
 * =========================================================
 */


export async function requestPlan(

    task,

    previousError = "",

    context = {}

) {


    /*
     * =====================================================
     * GROQ CLIENT
     * =====================================================
     */


    const groq =
        getGroqClient();


    /*
     * =====================================================
     * SYSTEM INSTRUCTIONS
     * =====================================================
     */


    const instructions =
        buildPlannerInstructions();


    /*
     * =====================================================
     * PLANNING CONTEXT
     * =====================================================
     */


    const planningContextText =
        buildPlanningContextText(
            context
        );


    /*
     * =====================================================
     * AVAILABLE TOOLS
     * =====================================================
     */


    const toolsText =
        buildToolsText();


    /*
     * =====================================================
     * RETRY CONTEXT
     * =====================================================
     */


    const retryContext =
        previousError
            ? [

                "",

                "ПРЕДЫДУЩИЙ ПЛАН БЫЛ ОТКЛОНЁН:",

                previousError,

                "",

                "Исправь ошибку и создай новый валидный план."

            ].join(
                "\n"
            )
            : "";


    /*
     * =====================================================
     * USER PROMPT
     * =====================================================
     */


    const userPromptParts =
        [

            "ЗАДАЧА:",

            String(
                task || ""
            ).trim()

        ];


    /*
     * PlanningContext добавляем
     * только если он действительно есть.
     */


    if (planningContextText) {

        userPromptParts.push(

            "",

            planningContextText

        );

    }


    /*
     * Доступные инструменты.
     */


    userPromptParts.push(

        "",

        "ДОСТУПНЫЕ ИНСТРУМЕНТЫ:",

        toolsText

    );


    /*
     * Контекст предыдущей ошибки Planner.
     */


    if (retryContext) {

        userPromptParts.push(
            retryContext
        );

    }


    /*
     * =====================================================
     * AI REQUEST
     * =====================================================
     */


    const response =
        await groq
            .chat
            .completions
            .create({

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

                        content:
                            userPromptParts.join(
                                "\n"
                            )

                    }

                ]

            });


    /*
     * =====================================================
     * RAW MODEL RESPONSE
     * =====================================================
     */


    return (

        response
            ?.choices
            ?.[0]
            ?.message
            ?.content || ""

    );

}
