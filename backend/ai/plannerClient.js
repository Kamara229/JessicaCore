/*
 * =========================================================
 * JESSICA PLANNER CLIENT v3
 * =========================================================
 *
 * AI transport Planner.
 *
 *
 * Ответственность:
 *
 * - model configuration;
 * - JSON output mode;
 * - message validation;
 * - Groq request;
 * - technical AI retry.
 *
 *
 * НЕ отвечает за:
 *
 * - prompt construction;
 * - PlanningContext;
 * - plan parsing;
 * - semantic plan validation;
 * - Execution Retry;
 * - Execution Replan.
 *
 *
 * КРИТИЧЕСКО:
 *
 * Provider/network failure
 * не должен автоматически становиться
 * Execution Replan.
 *
 * =========================================================
 */


import {
    getGroqClient
} from "./groqClient.js";


import {
    executeAIWithRetry
} from "./aiRetry.js";


/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


const PLANNER_MODEL =
    "openai/gpt-oss-20b";


const PLANNER_MAX_COMPLETION_TOKENS =
    800;


const PLANNER_REASONING_EFFORT =
    "low";


const PLANNER_AI_ATTEMPTS =
    3;


/*
 * =========================================================
 * MESSAGE VALIDATION
 * =========================================================
 */


function validateMessages(
    messages
) {

    if(
        !Array.isArray(
            messages
        )
    ){

        throw new Error(
            "Planner messages должны быть массивом"
        );

    }


    if(
        messages.length === 0
    ){

        throw new Error(
            "Planner messages пустые"
        );

    }


    for(
        let index = 0;

        index < messages.length;

        index++
    ){

        const message =
            messages[index];


        if(
            !message
            ||
            typeof message !== "object"
            ||
            Array.isArray(message)
        ){

            throw new Error(

                `Planner message ${index + 1} некорректен`

            );

        }


        if(
            typeof message.role !== "string"
            ||
            !message.role.trim()
        ){

            throw new Error(

                `Planner message ${index + 1} не содержит role`

            );

        }


        if(
            typeof message.content !== "string"
            ||
            !message.content.trim()
        ){

            throw new Error(

                `Planner message ${index + 1} не содержит content`

            );

        }

    }

}


/*
 * =========================================================
 * REQUEST OPTIONS
 * =========================================================
 */


function buildPlannerRequestOptions(
    messages
) {

    return {

        model:
            PLANNER_MODEL,


        /*
         * Planner должен быть максимально
         * детерминированным.
         */


        temperature:
            0,


        /*
         * Низкий reasoning budget.
         */


        reasoning_effort:
            PLANNER_REASONING_EFFORT,


        /*
         * Внутреннее reasoning
         * не возвращаем.
         */


        include_reasoning:
            false,


        /*
         * Ограничиваем JSON Plan.
         */


        max_completion_tokens:
            PLANNER_MAX_COMPLETION_TOKENS,


        /*
         * Только JSON object.
         */


        response_format: {

            type:
                "json_object"

        },


        /*
         * Planner описывает tools в JSON,
         * но сам tools НЕ вызывает.
         *
         * Явно фиксируем контракт.
         */


        tool_choice:
            "none",


        messages

    };

}


/*
 * =========================================================
 * PLANNER CHAT
 * =========================================================
 */


export async function plannerChat(
    messages
) {

    validateMessages(
        messages
    );


    const groq =

        getGroqClient();


    const requestOptions =

        buildPlannerRequestOptions(
            messages
        );


    /*
     * =====================================================
     * TECHNICAL AI RETRY
     * =====================================================
     *
     * Здесь ловятся:
     *
     * - 429 TPM/RPM;
     * - 408;
     * - 5xx;
     * - network;
     * - известный Groq GPT-OSS tool-call glitch.
     *
     *
     * Semantic invalid Plan сюда не попадает.
     *
     * =====================================================
     */


    return await executeAIWithRetry(

        async () => {

            return await groq
                .chat
                .completions
                .create(
                    requestOptions
                );

        },

        {

            label:
                "Jessica Planner AI",

            maxAttempts:
                PLANNER_AI_ATTEMPTS

        }

    );

}


/*
 * =========================================================
 * MODEL INFO
 * =========================================================
 */


export function getPlannerModel()
{

    return PLANNER_MODEL;

}


/*
 * =========================================================
 * OUTPUT LIMIT INFO
 * =========================================================
 */


export function getPlannerMaxCompletionTokens()
{

    return PLANNER_MAX_COMPLETION_TOKENS;

}
