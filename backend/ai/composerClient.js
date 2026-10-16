/*
 * =========================================================
 * JESSICA COMPOSER CLIENT v1
 * =========================================================
 *
 * AI transport обычного Answer Composer.
 *
 * Здесь находятся:
 *
 * - Groq client;
 * - model;
 * - generation parameters;
 * - technical AI retry.
 *
 * Бизнес-логики Composer здесь нет.
 * =========================================================
 */


import {
    getGroqClient
} from "./groqClient.js";


import {
    executeAIWithRetry
} from "./aiRetry.js";


const COMPOSER_MODEL =
    "openai/gpt-oss-20b";


const COMPOSER_TEMPERATURE =
    0.2;


const COMPOSER_AI_ATTEMPTS =
    3;


/*
 * =========================================================
 * AVAILABILITY
 * =========================================================
 */


export function isComposerClientAvailable()
{

    return Boolean(
        process.env.GROQ_API_KEY
    );

}


/*
 * =========================================================
 * REQUEST
 * =========================================================
 */


export async function requestComposerAI({

    instructions,

    input

} = {}) {

    if(
        !isComposerClientAvailable()
    ){

        throw new Error(
            "Groq Answer Composer не настроен"
        );

    }


    if(
        typeof instructions !== "string"
        ||
        !instructions.trim()
    ){

        throw new Error(
            "Answer Composer instructions пустые"
        );

    }


    if(
        typeof input !== "string"
        ||
        !input.trim()
    ){

        throw new Error(
            "Answer Composer input пустой"
        );

    }


    const groq =
        getGroqClient();


    return await executeAIWithRetry(

        () =>
            groq
                .chat
                .completions
                .create({

                    model:
                        COMPOSER_MODEL,

                    temperature:
                        COMPOSER_TEMPERATURE,

                    messages: [

                        {
                            role:
                                "system",

                            content:
                                instructions.trim()
                        },

                        {
                            role:
                                "user",

                            content:
                                input.trim()
                        }

                    ]

                }),

        {

            label:
                "Jessica Answer Composer AI",

            maxAttempts:
                COMPOSER_AI_ATTEMPTS

        }

    );

}


/*
 * =========================================================
 * MODEL INFO
 * =========================================================
 */


export function getComposerModel()
{

    return COMPOSER_MODEL;

}
