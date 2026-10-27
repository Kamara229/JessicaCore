/*
 * =========================================================
 * JESSICA PATTERN EXTRACTION CLIENT v1
 * =========================================================
 *
 * AI transport автономного
 * Experience Pattern Extraction.
 *
 *
 * Ответственность:
 *
 * - Groq client;
 * - выбранная модель;
 * - AI request;
 * - technical retry.
 *
 *
 * НЕ:
 *
 * - строит Pattern;
 * - валидирует Pattern;
 * - анализирует Execution Evidence;
 * - сохраняет Experience.
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


const DEFAULT_PATTERN_MODEL =
    "openai/gpt-oss-20b";


const PATTERN_AI_ATTEMPTS =
    3;


/*
 * =========================================================
 * MODEL
 * =========================================================
 */


function resolvePatternModel()
{

    return (

        process.env.JESSICA_LEARNING_MODEL

        ||

        process.env.GROQ_MODEL

        ||

        DEFAULT_PATTERN_MODEL

    );

}


/*
 * =========================================================
 * AVAILABILITY
 * =========================================================
 */


export function isPatternExtractionAvailable()
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


export async function requestPatternExtractionAI({

    systemPrompt,

    evidenceText,

    signal = null,
    client = null,
    retry = executeAIWithRetry

} = {}) {

    if(
        !client && !isPatternExtractionAvailable()
    ){

        throw new Error(
            "Groq Pattern Extractor не настроен"
        );

    }


    if(
        typeof systemPrompt !== "string"
        ||
        !systemPrompt.trim()
    ){

        throw new Error(
            "Pattern Extractor system prompt пустой"
        );

    }


    if(
        typeof evidenceText !== "string"
        ||
        !evidenceText.trim()
    ){

        throw new Error(
            "Pattern Extractor evidence пустой"
        );

    }


    const groq =
        client || getGroqClient();
    if(signal?.aborted){
        const error=new Error("Pattern AI request aborted");
        error.name="AbortError";
        throw error;
    }


    return await retry(

        async () => {

            return await groq
                .chat
                .completions
                .create({

                    model:
                        resolvePatternModel(),

                    temperature:
                        0.1,

                    messages: [

                        {

                            role:
                                "system",

                            content:
                                systemPrompt.trim()

                        },

                        {

                            role:
                                "user",

                            content:

                                "Execution Evidence:\n"

                                +

                                evidenceText

                        }

                    ]

                },{signal});

        },

        {

            signal,
            label:
                "Jessica Pattern Extractor AI",

            maxAttempts:
                PATTERN_AI_ATTEMPTS

        }

    );

}


/*
 * =========================================================
 * MODEL INFO
 * =========================================================
 */


export function getPatternExtractionModel()
{

    return resolvePatternModel();

}
