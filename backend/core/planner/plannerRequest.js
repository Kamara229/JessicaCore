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
 * Отвечает только за формирование запроса
 * к AI Planner.
 *
 *
 * Цепочка:
 *
 * task
 *   +
 * PlanningContext
 *   +
 * Tools
 *   +
 * Retry Error
 *        ↓
 *      GPT Planner
 *
 *
 * НЕ отвечает за:
 *
 * - parsing;
 * - normalization;
 * - validation;
 * - retry цикл;
 * - Experience поиск;
 * - выполнение инструментов.
 *
 * =========================================================
 */


/*
 * =========================================================
 * MODEL CONFIG
 * =========================================================
 */


const PLANNER_MODEL =
    "openai/gpt-oss-20b";


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
 * BUILD RETRY CONTEXT
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

    ].join(
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
     * GROQ CLIENT
     * =====================================================
     */


    const groq =
        getGroqClient();



    /*
     * =====================================================
     * SYSTEM PROMPT
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
     * USER MESSAGE
     * =====================================================
     */


    const userPromptParts =
        [];



    userPromptParts.push(

        "=== ТЕКУЩАЯ ЗАДАЧА ===",

        cleanTask

    );



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
        await groq
            .chat
            .completions
            .create({

                model:
                    PLANNER_MODEL,


                temperature:
                    0,


                messages:
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

            });



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
