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
 * AI-запрос для анализа исправления пользователя.
 *
 *
 * Теперь используется:
 *
 * Groq Structured Outputs
 *
 * response_format:
 * json_schema
 * strict: true
 *
 *
 * Это гарантирует:
 *
 * - валидный JSON;
 * - обязательные поля;
 * - отсутствие лишних полей;
 * - стабильную структуру ответа.
 *
 *
 * Этот файл НЕ:
 *
 * - парсит JSON;
 * - валидирует смысл;
 * - создаёт Skill;
 * - сохраняет Experience;
 * - подтверждает обучение.
 *
 * =========================================================
 */


/*
 * =========================================================
 * MODEL
 * =========================================================
 */


const LEARNING_ANALYZER_MODEL =
    "openai/gpt-oss-20b";


/*
 * =========================================================
 * NORMALIZE TEXT
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
 * REQUEST LEARNING ANALYSIS
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
     * INPUT
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
     * USER PROMPT
     * =====================================================
     */


    const userPromptParts =
        [

            "ИСХОДНАЯ ЗАДАЧА:",

            cleanTask

        ];


    if (cleanPreviousAnswer) {

        userPromptParts.push(

            "",

            "ПРЕДЫДУЩИЙ ОТВЕТ JESSICA:",

            cleanPreviousAnswer

        );

    }


    userPromptParts.push(

        "",

        "ИСПРАВЛЕНИЕ ПОЛЬЗОВАТЕЛЯ:",

        cleanCorrection

    );


    if (cleanCorrectedAnswer) {

        userPromptParts.push(

            "",

            "ПРАВИЛЬНЫЙ ОТВЕТ:",

            cleanCorrectedAnswer

        );

    }


    userPromptParts.push(

        "",

        "Проанализируй исправление пользователя и сформируй Learning Proposal."

    );


    /*
     * =====================================================
     * GROQ
     * =====================================================
     */


    const groq =
        getGroqClient();


    /*
     * =====================================================
     * AI REQUEST WITH STRUCTURED OUTPUT
     * =====================================================
     */


    const response =
        await groq
            .chat
            .completions
            .create({

                model:
                    LEARNING_ANALYZER_MODEL,


                temperature:
                    0,


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
                            userPromptParts.join(
                                "\n"
                            )

                    }

                ]

            });


    /*
     * =====================================================
     * RESPONSE
     * =====================================================
     *
     * Structured Output всё равно
     * возвращает JSON-строку в content.
     *
     * Поэтому оставляем старый контракт:
     *
     * requestLearningAnalysis()
     *       ↓
     * parser
     *
     * =====================================================
     */


    return (

        response
            ?.choices
            ?.[0]
            ?.message
            ?.content || ""

    );

}
