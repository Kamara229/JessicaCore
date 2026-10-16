/*
 * =========================================================
 * JESSICA COMPLEX COMPOSER CLIENT v1
 * =========================================================
 *
 * AI transport для Complex Answer Composer.
 *
 *
 * Ответственность:
 *
 * - модель;
 * - AI request;
 * - technical retry.
 *
 *
 * НЕ:
 *
 * - форматирует результаты subtasks;
 * - создаёт fallback;
 * - определяет статус;
 * - изменяет результаты;
 * - решает пользовательскую задачу.
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


const COMPLEX_COMPOSER_MODEL =
    "openai/gpt-oss-20b";


const COMPLEX_COMPOSER_AI_ATTEMPTS =
    3;


/*
 * =========================================================
 * AVAILABILITY
 * =========================================================
 */


export function isComplexComposerAvailable()
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


export async function requestComplexComposerAnswer({

    instructions,

    input

} = {}) {

    /*
     * =====================================================
     * AVAILABILITY
     * =====================================================
     */


    if(
        !isComplexComposerAvailable()
    ){

        throw new Error(
            "Groq Complex Composer не настроен"
        );

    }


    /*
     * =====================================================
     * INPUT
     * =====================================================
     */


    if(
        typeof instructions !== "string"
        ||
        !instructions.trim()
    ){

        throw new Error(
            "Complex Composer instructions пустые"
        );

    }


    if(
        typeof input !== "string"
        ||
        !input.trim()
    ){

        throw new Error(
            "Complex Composer input пустой"
        );

    }


    /*
     * =====================================================
     * CLIENT
     * =====================================================
     */


    const groq =
        getGroqClient();


    /*
     * =====================================================
     * REQUEST
     * =====================================================
     */


    return await executeAIWithRetry(

        async () => {

            return await groq
                .responses
                .create({

                    model:
                        COMPLEX_COMPOSER_MODEL,

                    instructions:
                        instructions.trim(),

                    input:
                        input.trim()

                });

        },

        {

            label:
                "Jessica Complex Composer AI",

            maxAttempts:
                COMPLEX_COMPOSER_AI_ATTEMPTS

        }

    );

}


/*
 * =========================================================
 * MODEL INFO
 * =========================================================
 */


export function getComplexComposerModel()
{

    return COMPLEX_COMPOSER_MODEL;

}
