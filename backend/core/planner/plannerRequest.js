import OpenAI from "openai";

import {
    buildPlannerInstructions
} from "./plannerPrompt.js";

import {
    buildToolsText
} from "./plannerTools.js";


/*
 * =========================================================
 * JESSICA PLANNER REQUEST
 * =========================================================
 *
 * Отвечает только за запрос к AI-модели Planner.
 *
 * Здесь находятся:
 *
 * - подключение Groq;
 * - модель Planner;
 * - сбор prompt;
 * - передача задачи;
 * - передача доступных инструментов;
 * - передача ошибки предыдущей попытки.
 *
 * Не содержит:
 *
 * - parsing JSON;
 * - нормализацию плана;
 * - валидацию плана;
 * - retry-цикл;
 * - Experience;
 * - выполнение инструментов.
 *
 * =========================================================
 */


/*
 * =========================================================
 * GROQ CLIENT
 * =========================================================
 */


const groq =
    new OpenAI({

        apiKey:
            process.env.GROQ_API_KEY,

        baseURL:
            "https://api.groq.com/openai/v1"

    });


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

    previousError = ""

) {


    /*
     * =====================================================
     * SYSTEM INSTRUCTIONS
     * =====================================================
     */


    const instructions =
        buildPlannerInstructions();



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
     *
     * Если предыдущий план был отклонён,
     * Planner получает причину ошибки.
     *
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

            ].join("\n")
            : "";



    /*
     * =====================================================
     * AI REQUEST
     * =====================================================
     */


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

                        String(
                            task || ""
                        ).trim(),

                        "",

                        "ДОСТУПНЫЕ ИНСТРУМЕНТЫ:",

                        toolsText,

                        retryContext

                    ].join(
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
