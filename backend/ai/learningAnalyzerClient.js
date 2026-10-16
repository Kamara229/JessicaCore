/*
 * =========================================================
 * JESSICA LEARNING ANALYZER CLIENT v1
 * =========================================================
 *
 * AI transport для Learning Analyzer.
 *
 *
 * Ответственность:
 *
 * - модель;
 * - generation parameters;
 * - Structured Output request;
 * - общий Groq client;
 * - technical AI retry.
 *
 *
 * НЕ:
 *
 * - строит Learning Prompt;
 * - анализирует correction;
 * - создаёт Skill;
 * - создаёт Proposal;
 * - сохраняет Experience;
 * - выполняет approval.
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


const LEARNING_ANALYZER_MODEL =
    "openai/gpt-oss-20b";


const LEARNING_ANALYZER_TEMPERATURE =
    0;


const LEARNING_ANALYZER_MAX_TOKENS =
    4000;


const LEARNING_ANALYZER_AI_ATTEMPTS =
    3;


/*
 * =========================================================
 * AVAILABILITY
 * =========================================================
 */


export function isLearningAnalyzerAvailable()
{

    return Boolean(
        process.env.GROQ_API_KEY
    );

}


/*
 * =========================================================
 * VALIDATE RESPONSE FORMAT
 * =========================================================
 */


function validateResponseFormat(
    responseFormat
) {

    if(
        !responseFormat
        ||
        typeof responseFormat !== "object"
        ||
        Array.isArray(responseFormat)
    ){

        throw new Error(
            "Learning Analyzer response format не указан"
        );

    }

}


/*
 * =========================================================
 * REQUEST
 * =========================================================
 */


export async function requestLearningAnalyzerAI({

    systemPrompt,

    userPrompt,

    responseFormat

} = {}) {

    /*
     * =====================================================
     * AVAILABILITY
     * =====================================================
     */


    if(
        !isLearningAnalyzerAvailable()
    ){

        throw new Error(
            "Groq Learning Analyzer не настроен"
        );

    }


    /*
     * =====================================================
     * INPUT
     * =====================================================
     */


    if(
        typeof systemPrompt !== "string"
        ||
        !systemPrompt.trim()
    ){

        throw new Error(
            "Learning Analyzer system prompt пустой"
        );

    }


    if(
        typeof userPrompt !== "string"
        ||
        !userPrompt.trim()
    ){

        throw new Error(
            "Learning Analyzer user prompt пустой"
        );

    }


    validateResponseFormat(
        responseFormat
    );


    /*
     * =====================================================
     * CLIENT
     * =====================================================
     */


    const groq =
        getGroqClient();


    /*
     * =====================================================
     * TECHNICAL AI RETRY
     * =====================================================
     *
     * Здесь повторяются только временные
     * provider / network failures.
     *
     * Semantic Learning logic сюда
     * не относится.
     *
     * =====================================================
     */


    return await executeAIWithRetry(

        async () => {

            return await groq
                .chat
                .completions
                .create({

                    model:
                        LEARNING_ANALYZER_MODEL,


                    temperature:
                        LEARNING_ANALYZER_TEMPERATURE,


                    max_completion_tokens:
                        LEARNING_ANALYZER_MAX_TOKENS,


                    response_format:
                        responseFormat,


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
                                userPrompt.trim()

                        }

                    ]

                });

        },

        {

            label:
                "Jessica Learning Analyzer AI",

            maxAttempts:
                LEARNING_ANALYZER_AI_ATTEMPTS

        }

    );

}


/*
 * =========================================================
 * MODEL INFO
 * =========================================================
 */


export function getLearningAnalyzerModel()
{

    return LEARNING_ANALYZER_MODEL;

}
