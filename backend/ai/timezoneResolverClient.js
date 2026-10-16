/*
 * =========================================================
 * JESSICA TIMEZONE RESOLVER CLIENT v1
 * =========================================================
 *
 * AI transport преобразования:
 *
 * human location
 *      ↓
 * IANA timezone
 *
 *
 * AI НЕ определяет текущее время.
 *
 * Он используется только как
 * language / geography resolver.
 *
 *
 * Само время вычисляет сервер.
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


const TIMEZONE_RESOLVER_MODEL =
    "openai/gpt-oss-20b";


const TIMEZONE_RESOLVER_AI_ATTEMPTS =
    3;


/*
 * =========================================================
 * AVAILABILITY
 * =========================================================
 */


export function isTimezoneResolverAvailable()
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


export async function requestTimezoneResolution({

    instructions,

    location

} = {}) {

    if(
        !isTimezoneResolverAvailable()
    ){

        throw new Error(
            "Groq Timezone Resolver не настроен"
        );

    }


    if(
        typeof instructions !== "string"
        ||
        !instructions.trim()
    ){

        throw new Error(
            "Timezone Resolver instructions пустые"
        );

    }


    if(
        typeof location !== "string"
        ||
        !location.trim()
    ){

        throw new Error(
            "Timezone Resolver location пустой"
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
                        TIMEZONE_RESOLVER_MODEL,

                    instructions:
                        instructions.trim(),

                    input:

                        `Местоположение: ${location.trim()}`,

                    reasoning: {

                        effort:
                            "low"

                    }

                });

        },

        {

            label:
                "Jessica Timezone Resolver AI",

            maxAttempts:
                TIMEZONE_RESOLVER_AI_ATTEMPTS

        }

    );

}


/*
 * =========================================================
 * MODEL INFO
 * =========================================================
 */


export function getTimezoneResolverModel()
{

    return TIMEZONE_RESOLVER_MODEL;

}
