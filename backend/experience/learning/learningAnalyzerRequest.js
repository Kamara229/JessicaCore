import {
    getGroqClient
} from "../../ai/groqClient.js";

import {
    LEARNING_ANALYZER_SYSTEM_PROMPT
} from "./learningAnalyzerPrompt.js";


/*
 * =========================================================
 * JESSICA LEARNING ANALYZER REQUEST
 * =========================================================
 *
 * Отвечает только за AI-запрос
 * для анализа исправления пользователя.
 *
 *
 * Вход:
 *
 * - исходная задача;
 * - предыдущий ответ Jessica;
 * - исправление пользователя;
 * - правильный ответ, если он известен.
 *
 *
 * Выход:
 *
 * - сырой текст ответа AI.
 *
 *
 * Обычно модель должна вернуть JSON,
 * но parsing и validation выполняются
 * отдельными модулями.
 *
 *
 * Этот файл НЕ:
 *
 * - парсит JSON;
 * - создаёт Learning Proposal;
 * - сохраняет Experience;
 * - подтверждает обучение;
 * - изменяет Skills;
 * - повторно выполняет задачу.
 *
 * =========================================================
 */


/*
 * =========================================================
 * MODEL
 * =========================================================
 *
 * Пока используем ту же модель,
 * что и Planner.
 *
 * Позже Learning может получить
 * отдельную более сильную модель,
 * не затрагивая Planner.
 *
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


    /*
     * Предыдущий ответ Jessica
     * может отсутствовать.
     */


    if (cleanPreviousAnswer) {

        userPromptParts.push(

            "",

            "ПРЕДЫДУЩИЙ ОТВЕТ JESSICA:",

            cleanPreviousAnswer

        );

    }


    /*
     * Исправление пользователя —
     * обязательная часть обучения.
     */


    userPromptParts.push(

        "",

        "ИСПРАВЛЕНИЕ ПОЛЬЗОВАТЕЛЯ:",

        cleanCorrection

    );


    /*
     * Если пользователь отдельно
     * дал правильный итоговый ответ,
     * передаём его отдельным блоком.
     */


    if (cleanCorrectedAnswer) {

        userPromptParts.push(

            "",

            "ПРАВИЛЬНЫЙ ОТВЕТ:",

            cleanCorrectedAnswer

        );

    }


    userPromptParts.push(

        "",

        "Проанализируй исправление и сформируй Learning Proposal согласно системной инструкции."

    );


    /*
     * =====================================================
     * GROQ CLIENT
     * =====================================================
     */


    const groq =
        getGroqClient();


    /*
     * =====================================================
     * AI REQUEST
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
     * RAW RESPONSE
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
