/*
 * =========================================================
 * JESSICA SOURCE SELECTOR CLIENT v1
 * =========================================================
 *
 * AI transport Source Selector.
 *
 *
 * Ответственность:
 *
 * - модель Source Selector;
 * - Groq request;
 * - technical AI retry.
 *
 *
 * НЕ:
 *
 * - оценивает источники;
 * - строит prompt;
 * - парсит JSON;
 * - выбирает result.
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


const SOURCE_SELECTOR_MODEL =
    "openai/gpt-oss-20b";


const SOURCE_SELECTOR_AI_ATTEMPTS =
    3;


/*
 * =========================================================
 * AVAILABILITY
 * =========================================================
 */


export function isSourceSelectorAvailable()
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


export async function requestSourceSelection({

    instructions,

    task,

    results

} = {}) {

    if(
        !isSourceSelectorAvailable()
    ){

        throw new Error(
            "Groq Source Selector не настроен"
        );

    }


    if(
        typeof instructions !== "string"
        ||
        !instructions.trim()
    ){

        throw new Error(
            "Source Selector instructions пустые"
        );

    }


    if(
        typeof task !== "string"
        ||
        !task.trim()
    ){

        throw new Error(
            "Source Selector task пустой"
        );

    }


    if(
        !Array.isArray(results)
        ||
        results.length === 0
    ){

        throw new Error(
            "Source Selector results пустые"
        );

    }


    const groq =
        getGroqClient();


    return await executeAIWithRetry(

        async () => {

            return await groq
                .chat
                .completions
                .create({

                    model:
                        SOURCE_SELECTOR_MODEL,

                    temperature:
                        0,

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

                            content: [

                                "ЗАДАЧА:",

                                task.trim(),

                                "",

                                "РЕЗУЛЬТАТЫ ПОИСКА:",

                                JSON.stringify(
                                    results,
                                    null,
                                    2
                                )

                            ]
                                .join("\n")

                        }

                    ]

                });

        },

        {

            label:
                "Jessica Source Selector AI",

            maxAttempts:
                SOURCE_SELECTOR_AI_ATTEMPTS

        }

    );

}


/*
 * =========================================================
 * MODEL INFO
 * =========================================================
 */


export function getSourceSelectorModel()
{

    return SOURCE_SELECTOR_MODEL;

}
