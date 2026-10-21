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
 * JESSICA PLANNER REQUEST v3
 * =========================================================
 *
 * Task
 *      +
 * Semantic Repair
 *      +
 * PlanningContext
 *      +
 * Tools
 *      ↓
 * Planner AI
 *
 *
 * Semantic Repair намеренно расположен
 * сразу после Task.
 *
 * Он не должен теряться после длинного
 * Experience / PlanningContext.
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


    if(
        !text
    ){

        return "";
    }


    if(
        text.length <= limit
    ){

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
 * SEMANTIC REPAIR BLOCK
 * =========================================================
 */


function buildRetryContext(
    previousError
) {

    const feedback =

        limitText(

            previousError,

            MAX_RETRY_LENGTH

        );


    if(
        !feedback
    ){

        return "";
    }


    return [

        "=== ОБЯЗАТЕЛЬНЫЙ РЕМОНТ ПРЕДЫДУЩЕЙ ПОПЫТКИ ===",

        "",

        (
            "Предыдущий Execution Plan был отклонён машинным Validator."
        ),

        (
            "Следующий Plan обязан исправить именно указанную semantic error."
        ),

        "",

        feedback,

        "",

        "ПРАВИЛА СЛЕДУЮЩЕЙ ПОПЫТКИ:",

        (
            "- не копируй отклонённые tool arguments;"
        ),

        (
            "- не маскируй старый Plan изменением reasoningSummary или step.id;"
        ),

        (
            "- если Validator запретил неподтверждённое конкретное значение, "
            +
            "не заменяй его другим предполагаемым значением;"
        ),

        (
            "- перестрой маршрут так, чтобы неизвестный факт был сначала найден, "
            +
            "а не предположен;"
        ),

        (
            "- новый Plan всё равно обязан соблюдать все системные Evidence, "
            +
            "Grounding и Source Content contracts."
        ),

        "",

        "Верни новый JSON Execution Plan."

    ]
        .join("\n");

}


/*
 * =========================================================
 * CONTEXT
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
 * USER PROMPT
 * =========================================================
 */


function buildUserPrompt({

    task,

    context,

    previousError

}) {

    const parts =
        [];


    /*
     * =====================================================
     * TASK
     * =====================================================
     */


    parts.push(

        [

            "=== ТЕКУЩАЯ ЗАДАЧА ===",

            task

        ]
            .join("\n")

    );


    /*
     * =====================================================
     * SEMANTIC REPAIR
     * =====================================================
     *
     * Намеренно ДО общего Context.
     * =====================================================
     */


    const retry =

        buildRetryContext(
            previousError
        );


    if(
        retry
    ){

        parts.push(
            retry
        );

    }


    /*
     * =====================================================
     * EXPERIENCE + CONTEXT
     * =====================================================
     */


    const contextText =

        buildContextBlock(
            context
        );


    if(
        contextText
    ){

        parts.push(

            [

                "=== КОНТЕКСТ JESSICA ===",

                contextText

            ]
                .join("\n")

        );

    }


    /*
     * =====================================================
     * TOOLS
     * =====================================================
     */


    parts.push(

        [

            "=== ДОСТУПНЫЕ ИНСТРУМЕНТЫ ===",

            buildToolsText()

        ]
            .join("\n")

    );


    return parts.join(
        "\n\n"
    );

}


/*
 * =========================================================
 * REQUEST
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


    if(
        !cleanTask
    ){

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

        buildPlannerInstructions(cleanTask);


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
     * AI
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
