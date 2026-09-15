import {
    getGroqClient
} from "../../ai/groqClient.js";

import {
    LEARNING_ANALYZER_SYSTEM_PROMPT
} from "./learningAnalyzerPrompt.js";

import {
    LEARNING_ANALYZER_RESPONSE_FORMAT
} from "./learningAnalyzerSchema.js";


/*
 * =========================================================
 * JESSICA LEARNING ANALYZER REQUEST
 * =========================================================
 *
 * Отвечает только за обращение к AI-модели.
 *
 *
 * Цепочка:
 *
 * correction
 *      ↓
 * Learning Analyzer Request
 *      ↓
 * Groq Structured Output
 *      ↓
 * JSON string
 *      ↓
 * Parser
 *
 *
 * НЕ отвечает за:
 *
 * - анализ смысла;
 * - validation;
 * - создание Skill;
 * - сохранение Experience;
 * - approval.
 *
 * =========================================================
 */


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


/*
 * =========================================================
 * NORMALIZE
 * =========================================================
 */


function normalizeText(
    value
) {

    return String(
        value || ""
    ).trim();

}


/*
 * =========================================================
 * BUILD USER PROMPT
 * =========================================================
 */


function buildLearningPrompt({

    task,

    previousAnswer,

    correction,

    correctedAnswer

}) {


    const parts =
        [

            "ИСХОДНАЯ ЗАДАЧА:",

            task

        ];


    if (previousAnswer) {

        parts.push(

            "",

            "ПРЕДЫДУЩИЙ ОТВЕТ JESSICA:",

            previousAnswer

        );

    }


    parts.push(

        "",

        "ИСПРАВЛЕНИЕ ПОЛЬЗОВАТЕЛЯ:",

        correction

    );


    if (correctedAnswer) {

        parts.push(

            "",

            "ПРАВИЛЬНЫЙ ОТВЕТ:",

            correctedAnswer

        );

    }


    parts.push(

        "",

        "Проанализируй исправление пользователя.",

        "Создай переиспользуемый Learning Proposal.",

        "Заполни все поля JSON Schema."

    );


    return parts.join(
        "\n"
    );

}


/*
 * =========================================================
 * REQUEST
 * =========================================================
 */


export async function requestLearningAnalysis({

    task,

    previousAnswer = "",

    correction,

    correctedAnswer = ""

} = {}) {


    /*
     * =====================================================
     * INPUT VALIDATION
     * =====================================================
     */


    const cleanTask =
        normalizeText(
            task
        );


    const cleanPreviousAnswer =
        normalizeText(
            previousAnswer
        );


    const cleanCorrection =
        normalizeText(
            correction
        );


    const cleanCorrectedAnswer =
        normalizeText(
            correctedAnswer
        );


    if (!cleanTask) {

        throw new Error(
            "Learning Analyzer: исходная задача не указана"
        );

    }


    if (!cleanCorrection) {

        throw new Error(
            "Learning Analyzer: исправление пользователя не указано"
        );

    }


    /*
     * =====================================================
     * GROQ
     * =====================================================
     */


    const groq =
        getGroqClient();


    const userPrompt =
        buildLearningPrompt({

            task:
                cleanTask,

            previousAnswer:
                cleanPreviousAnswer,

            correction:
                cleanCorrection,

            correctedAnswer:
                cleanCorrectedAnswer

        });



    try {


        /*
         * =================================================
         * STRUCTURED OUTPUT REQUEST
         * =================================================
         */


        const response =
            await groq
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
                        LEARNING_ANALYZER_RESPONSE_FORMAT,


                    messages: [

                        {

                            role:
                                "system",

                            content:
                                LEARNING_ANALYZER_SYSTEM_PROMPT

                        },

                        {

                            role:
                                "user",

                            content:
                                userPrompt

                        }

                    ]

                });



        const content =
            response
                ?.choices
                ?.[0]
                ?.message
                ?.content;


        if (!content) {

            throw new Error(
                "Learning Analyzer: AI вернул пустой ответ"
            );

        }


        return content;


    } catch (error) {


        console.error(
            "Jessica Learning Analyzer request failed:",
            {

                message:
                    error?.message,

                code:
                    error?.code,

                type:
                    error?.type

            }
        );


        throw error;

    }

}
