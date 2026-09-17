/*
 * =========================================================
 * JESSICA COMPOSER REQUEST
 * =========================================================
 *
 * AI transport layer Answer Composer.
 *
 *
 * Отвечает только за:
 *
 * - создание AI client;
 * - модель Composer;
 * - системный prompt;
 * - AI request;
 * - retry AI-запроса.
 *
 *
 * НЕ отвечает за:
 *
 * - подготовку factual context;
 * - direct tool answer;
 * - parsing бизнес-результата;
 * - validation;
 * - execution;
 * - semantic outcome.
 *
 * =========================================================
 */


import OpenAI from "openai";

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


/*
 * =========================================================
 * AI CLIENT
 * =========================================================
 */


const groq =
    process.env.GROQ_API_KEY

        ? new OpenAI({

            apiKey:
                process.env.GROQ_API_KEY,

            baseURL:
                "https://api.groq.com/openai/v1"

        })

        : null;


/*
 * =========================================================
 * AVAILABILITY
 * =========================================================
 */


export function isComposerAvailable() {

    return Boolean(
        groq
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

    if (
        typeof input !== "string" ||
        !input.trim()
    ) {

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

    validateComposerInput(
        input
    );


    if (!groq) {

        throw new Error(
            "Groq Answer Composer не настроен"
        );

    }


    const instructions =
        getComposerInstructions();


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
                "Answer Composer"

        }

    );

}


/*
 * =========================================================
 * MODEL INFO
 * =========================================================
 */


export function getComposerModel() {

    return COMPOSER_MODEL;

}
