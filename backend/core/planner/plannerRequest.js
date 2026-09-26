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


import {
    normalizePlanningContext
} from "./planningContext.js";



/*
 * =========================================================
 * JESSICA PLANNER REQUEST v2
 * =========================================================
 *
 * Формирует запрос AI Planner.
 *
 *
 * Flow:
 *
 * Task
 *   +
 * Normalized PlanningContext
 *   +
 * Experience Memory
 *   +
 * Tools
 *   +
 * Retry Feedback
 *
 *        ↓
 *
 * Planner AI
 *
 *
 * Отвечает только за:
 *
 * - сбор system prompt;
 * - сбор user prompt;
 * - добавление контекста;
 * - передачу ошибки предыдущей попытки.
 *
 *
 * НЕ:
 *
 * - создаёт план;
 * - парсит JSON;
 * - валидирует план;
 * - выполняет инструменты;
 * - изменяет Experience.
 *
 * =========================================================
 */





/*
 * =========================================================
 * LIMITS
 * =========================================================
 */


const MAX_TASK_LENGTH =
    8000;


const MAX_CONTEXT_LENGTH =
    6000;


const MAX_RETRY_LENGTH =
    3000;







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
 * LIMIT TEXT
 * =========================================================
 */


function limitText(
    value,
    limit
) {


    const text =
        safeString(
            value
        );



    if (!text) {

        return "";

    }



    if (
        text.length <= limit
    ) {

        return text;

    }



    return (

        text.slice(
            0,
            limit
        )

        +

        "\n\n[Контекст сокращён]"

    );

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
        limitText(
            previousError,
            MAX_RETRY_LENGTH
        );



    if (!error) {

        return "";

    }



    return [

        "=== ОБРАТНАЯ СВЯЗЬ ПРЕДЫДУЩЕЙ ПОПЫТКИ ===",

        "",

        error,

        "",

        "Исправь маршрут.",

        "Не повторяй ошибочный план без изменений."

    ]
    .join("\n");

}







/*
 * =========================================================
 * BUILD CONTEXT
 * =========================================================
 */


function buildContextBlock(
    context
) {


    const normalized =

        normalizePlanningContext(
            context
        );



    const text =

        buildPlanningContextText(
            normalized
        );



    return limitText(

        text,

        MAX_CONTEXT_LENGTH

    );


}







/*
 * =========================================================
 * BUILD USER PROMPT
 * =========================================================
 */


function buildUserPrompt({

    task,

    context,

    previousError

}) {


    const parts = [];






    /*
     * TASK
     */


    parts.push(

        [

            "=== ТЕКУЩАЯ ЗАДАЧА ===",

            task

        ]
        .join("\n")

    );







    /*
     * EXPERIENCE + CONTEXT
     */


    const contextText =

        buildContextBlock(
            context
        );



    if (
        contextText
    ) {


        parts.push(

            [

                "=== КОНТЕКСТ JESSICA ===",

                contextText

            ]
            .join("\n")

        );

    }








    /*
     * TOOLS
     */


    parts.push(

        [

            "=== ДОСТУПНЫЕ ИНСТРУМЕНТЫ ===",

            buildToolsText()

        ]
        .join("\n")

    );








    /*
     * RETRY
     */


    const retry =

        buildRetryContext(
            previousError
        );



    if (
        retry
    ) {


        parts.push(
            retry
        );

    }







    return parts.join(

        "\n\n"

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


    const cleanTask =

        limitText(

            safeString(
                task
            ),

            MAX_TASK_LENGTH

        );



    if (!cleanTask) {

        throw new Error(
            "Planner task пустой"
        );

    }





    /*
     * =====================================================
     * SYSTEM PROMPT
     * =====================================================
     */


    const instructions =

        buildPlannerInstructions();







    /*
     * =====================================================
     * USER PROMPT
     * =====================================================
     */


    const userPrompt =

        buildUserPrompt({

            task:
                cleanTask,

            context,

            previousError

        });







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
