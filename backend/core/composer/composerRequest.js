/*
 * =========================================================
 * JESSICA COMPOSER REQUEST v2
 * =========================================================
 *
 * AI transport layer Answer Composer.
 *
 *
 * Ответственность:
 *
 * - модель Composer;
 * - системный prompt;
 * - AI request;
 * - technical AI retry.
 *
 *
 * Общий Groq client:
 *
 * backend/ai/groqClient.js
 *
 *
 * НЕ:
 *
 * - создаёт собственный OpenAI client;
 * - подготавливает factual context;
 * - выполняет tools;
 * - валидирует результат;
 * - определяет semantic outcome.
 *
 * =========================================================
 */


import {
    getGroqClient
} from "../../ai/groqClient.js";


import {
    executeAIWithRetry
} from "../../ai/aiRetry.js";


import {
    getComposerInstructions
} from "./composerPrompt.js";


/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


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


export function isComposerAvailable()
{

    return Boolean(
        process.env.GROQ_API_KEY
    );

}


/*
 * =========================================================
 * VALIDATE INPUT
 * =========================================================
 */


function validateComposerInput(
    input
) {

    if(
        typeof input !== "string"
        ||
        !input.trim()
    ){

        throw new Error(
            "Answer Composer получил пустой контекст"
        );

    }

}


/*
 * =========================================================
 * REQUEST
 * =========================================================
 */


export async function requestComposerAnswer(
    input
) {

    /*
     * =====================================================
     * INPUT
     * =====================================================
     */


    validateComposerInput(
        input
    );


    /*
     * =====================================================
     * AVAILABILITY
     * =====================================================
     */


    if(
        !isComposerAvailable()
    ){

        throw new Error(
            "Groq Answer Composer не настроен"
        );

    }


    /*
     * =====================================================
     * CLIENT
     * =====================================================
     */


    const groq =
        getGroqClient();


    const instructions =
        getComposerInstructions();


    /*
     * =====================================================
     * AI REQUEST
     * =====================================================
     */


    return await executeAIWithRetry(

        async () => {

            return await groq
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
                                instructions

                        },

                        {

                            role:
                                "user",

                            content:
                                input

                        }

                    ]

                });

        },

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
