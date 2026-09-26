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
 * Формирование запроса AI Planner.
 *
 *
 * Flow:
 *
 * Task
 *   +
 * Planning Context
 *   +
 * Experience Memory
 *   +
 * Tools
 *        ↓
 * Planner AI
 *
 *
 * Отвечает:
 *
 * - system prompt;
 * - user prompt;
 * - context;
 * - tools;
 * - retry feedback.
 *
 *
 * НЕ:
 *
 * - создаёт план;
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


const MAX_CONTEXT_LENGTH =
    6000;



const MAX_EXPERIENCE_LENGTH =
    2500;







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



    if (
        !text
    ) {

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
        safeString(
            previousError
        );



    if (
        !error
    ) {

        return "";

    }



    return [

        "=== ПРЕДЫДУЩАЯ ОШИБКА PLANNER ===",

        error,


        "",


        "Исправь ошибку. " +
        "Создай новый корректный JSON-план."

    ]
    .join(
        "\n"
    );

}








/*
 * =========================================================
 * EXPERIENCE INFO
 * =========================================================
 */


function buildExperienceMeta(
    context
) {


    const experience =
        context?.experience;



    if (
        !experience ||
        experience.available !== true
    ) {


        return "";

    }





    return [

        "=== EXPERIENCE MEMORY ===",


        `Источник:
${experience.source || "unknown"}`,


        `Уверенность:
${experience.confidence || 0}`,


        `Количество навыков:
${experience.skills?.length || 0}`


    ]
    .join(
        "\n"
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
        safeString(
            task
        );





    if (
        !cleanTask
    ) {


        throw new Error(
            "Planner task пустой"
        );

    }








    /*
     * =====================================================
     * SYSTEM
     * =====================================================
     */


    const instructions =
        buildPlannerInstructions();








    /*
     * =====================================================
     * CONTEXT
     * =====================================================
     */


    let planningContextText =
        buildPlanningContextText(
            context
        );



    planningContextText =
        limitText(

            planningContextText,

            MAX_CONTEXT_LENGTH

        );





    const experienceMeta =
        limitText(

            buildExperienceMeta(
                context
            ),

            MAX_EXPERIENCE_LENGTH

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


    const parts =
        [

            "=== ТЕКУЩАЯ ЗАДАЧА ===",

            cleanTask


        ];







    if (
        experienceMeta
    ) {


        parts.push(

            "",

            experienceMeta

        );

    }







    if (
        planningContextText
    ) {


        parts.push(

            "",


            "=== КОНТЕКСТ ПЛАНИРОВАНИЯ JESSICA ===",


            planningContextText

        );


    }







    parts.push(

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


        parts.push(

            "",

            retryContext

        );


    }







    const userPrompt =
        parts.join(
            "\n"
        );









    /*
     * =====================================================
     * AI CALL
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
