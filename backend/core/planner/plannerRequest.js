import {
    plannerChat
} from "../../ai/plannerClient.js";

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
 * Формирует запрос для AI Planner.
 *
 *
 * Отвечает только за:
 *
 * - сбор system prompt;
 * - сбор user prompt;
 * - добавление PlanningContext;
 * - добавление Tools;
 * - передачу ошибки предыдущей попытки.
 *
 *
 * НЕ отвечает за:
 *
 * - создание AI клиента;
 * - выбор модели;
 * - parsing;
 * - normalization;
 * - validation;
 * - retry цикл;
 * - Experience;
 * - выполнение инструментов.
 *
 *
 * Архитектура:
 *
 * task
 *   +
 * PlanningContext
 *   +
 * Tools
 *   +
 * Retry Error
 *
 *        ↓
 *
 * plannerClient
 *
 *        ↓
 *
 * AI Planner
 *
 * =========================================================
 */


/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


const MAX_CONTEXT_LENGTH =
    6000;


/*
 * =========================================================
 * SAFE STRING
 * =========================================================
 */


function safeString(
    value
) {

    return String(
        value || ""
    )
        .trim();

}



/*
 * =========================================================
 * RETRY CONTEXT
 * =========================================================
 */


function buildRetryContext(
    previousError
) {


    const error =
        safeString(
            previousError
        );


    if (!error) {

        return "";

    }


    return [

        "=== ПРЕДЫДУЩАЯ ОШИБКА PLANNER ===",

        error,

        "",

        "Исправь ошибку и создай новый валидный JSON-план."

    ]
    .join(
        "\n"
    );

}



/*
 * =========================================================
 * LIMIT CONTEXT
 * =========================================================
 */


function limitContext(
    value
) {


    const text =
        safeString(
            value
        );


    if (!text) {

        return "";

    }


    if (
        text.length <= MAX_CONTEXT_LENGTH
    ) {

        return text;

    }


    return (

        text.slice(
            0,
            MAX_CONTEXT_LENGTH
        )

        +

        "\n\n[Контекст сокращён]"

    );

}



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
     * INPUT
     * =====================================================
     */


    const cleanTask =
        safeString(
            task
        );


    if (!cleanTask) {

        throw new Error(
            "Planner task пустой"
        );

    }



    /*
     * =====================================================
     * SYSTEM INSTRUCTIONS
     * =====================================================
     */


    const instructions =
        buildPlannerInstructions();



    /*
     * =====================================================
     * EXPERIENCE CONTEXT
     * =====================================================
     */


    const planningContextText =
        limitContext(

            buildPlanningContextText(
                context
            )

        );



    /*
     * =====================================================
     * TOOLS
     * =====================================================
     */


    const toolsText =
        buildToolsText();



    /*
     * =====================================================
     * USER PROMPT
     * =====================================================
     */


    const userPromptParts =
        [

            "=== ТЕКУЩАЯ ЗАДАЧА ===",

            cleanTask

        ];



    if (
        planningContextText
    ) {

        userPromptParts.push(

            "",

            "=== НАКОПЛЕННЫЙ КОНТЕКСТ JESSICA ===",

            planningContextText

        );

    }



    userPromptParts.push(

        "",

        "=== ДОСТУПНЫЕ ИНСТРУМЕНТЫ ===",

        toolsText

    );



    const retryContext =
        buildRetryContext(
            previousError
        );


    if (
        retryContext
    ) {

        userPromptParts.push(

            "",

            retryContext

        );

    }



    const userPrompt =
        userPromptParts.join(
            "\n"
        );



    /*
     * =====================================================
     * AI REQUEST
     * =====================================================
     */


    const response =
        await plannerChat(

            [

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
                        userPrompt

                }

            ]

        );



    /*
     * =====================================================
     * RESPONSE
     * =====================================================
     */


    return (

        response
            ?.choices
            ?.[0]
            ?.message
            ?.content
            ||
            ""

    );

}
