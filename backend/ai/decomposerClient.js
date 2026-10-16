/*
 * =========================================================
 * JESSICA DECOMPOSER CLIENT v1
 * =========================================================
 *
 * AI transport Task Decomposer.
 *
 *
 * Ответственность:
 *
 * - модель Decomposer;
 * - параметры generation;
 * - Groq request;
 * - технический AI retry.
 *
 *
 * НЕ:
 *
 * - строит decomposition prompt;
 * - нормализует Memory;
 * - парсит JSON;
 * - валидирует subtasks;
 * - создаёт fallback.
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


const DECOMPOSER_MODEL =
    "openai/gpt-oss-20b";


const DECOMPOSER_REASONING_EFFORT =
    "medium";


const DECOMPOSER_AI_ATTEMPTS =
    3;


/*
 * =========================================================
 * AVAILABILITY
 * =========================================================
 */


export function isDecomposerAvailable()
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


export async function requestDecomposition({

    instructions,

    task,

    memory

} = {}) {

    if(
        !isDecomposerAvailable()
    ){

        throw new Error(
            "Groq Task Decomposer не настроен"
        );

    }


    if(
        typeof instructions !== "string"
        ||
        !instructions.trim()
    ){

        throw new Error(
            "Task Decomposer instructions пустые"
        );

    }


    if(
        typeof task !== "string"
        ||
        !task.trim()
    ){

        throw new Error(
            "Task Decomposer task пустой"
        );

    }


    const groq =
        getGroqClient();


    return await executeAIWithRetry(

        async () => {

            return await groq
                .responses
                .create({

                    model:
                        DECOMPOSER_MODEL,


                    instructions:
                        instructions.trim(),


                    input:

                        JSON.stringify({

                            task:
                                task.trim(),

                            memory:

                                Array.isArray(memory)

                                    ? memory

                                    : []

                        }),


                    reasoning: {

                        effort:
                            DECOMPOSER_REASONING_EFFORT

                    }

                });

        },

        {

            label:
                "Jessica Task Decomposer AI",

            maxAttempts:
                DECOMPOSER_AI_ATTEMPTS

        }

    );

}


/*
 * =========================================================
 * MODEL INFO
 * =========================================================
 */


export function getDecomposerModel()
{

    return DECOMPOSER_MODEL;

}
