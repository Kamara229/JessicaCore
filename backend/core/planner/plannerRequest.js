import OpenAI from "openai";

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
 * Отвечает только за запрос к AI-модели Planner.
 *
 * Здесь находятся:
 *
 * - подключение Groq;
 * - модель Planner;
 * - сбор prompt;
 * - передача задачи;
 * - передача дополнительного PlanningContext;
 * - передача доступных инструментов;
 * - передача ошибки предыдущей попытки.
 *
 *
 * Не содержит:
 *
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

    previousError = "",

    context = {}

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
     * PLANNING CONTEXT
     * =====================================================
     *
     * Сейчас context может быть пустым.
     *
     * Позже сюда будут передаваться:
     *
     * - Experience Jessica;
     * - успешные алгоритмы;
     * - правила источников;
     * - ограничения задачи;
     * - инструкции Earnings;
     * - другой дополнительный контекст.
     *
     * Если контекста нет,
     * buildPlanningContextText()
     * вернёт пустую строку.
     *
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

            ].join(
                "\n"
            )
            : "";



    /*
     * =====================================================
     * USER PROMPT
     * =====================================================
     *
     * Формируем prompt из независимых блоков.
     *
     * Пустой PlanningContext
     * вообще не добавляется.
     *
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
     * Дополнительный контекст
     * добавляем только если он существует.
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
     * Ошибка предыдущего плана.
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
