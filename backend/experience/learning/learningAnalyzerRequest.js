import {
    requestLearningAnalyzerAI
} from "../../ai/learningAnalyzerClient.js";


import {
    LEARNING_ANALYZER_SYSTEM_PROMPT
} from "./learningAnalyzerPrompt.js";


import {
    LEARNING_ANALYZER_RESPONSE_FORMAT
} from "./learningAnalyzerSchema.js";


/*
 * =========================================================
 * JESSICA LEARNING ANALYZER REQUEST v2
 * =========================================================
 *
 * Подготавливает semantic input
 * для Learning Analyzer.
 *
 *
 * Flow:
 *
 * correction
 *      ↓
 * Learning Analyzer Request
 *      ↓
 * Learning Analyzer AI Client
 *      ↓
 * Structured Output
 *      ↓
 * JSON string
 *      ↓
 * Parser
 *
 *
 * AI transport:
 *
 * backend/ai/learningAnalyzerClient.js
 *
 *
 * НЕ:
 *
 * - создаёт Groq client;
 * - управляет technical retry;
 * - анализирует смысл результата;
 * - валидирует Proposal;
 * - создаёт Skill;
 * - сохраняет Experience;
 * - выполняет approval.
 *
 * =========================================================
 */


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
    )
        .trim();

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

    const parts = [

        "ИСХОДНАЯ ЗАДАЧА:",

        task

    ];


    /*
     * =====================================================
     * PREVIOUS ANSWER
     * =====================================================
     */


    if(
        previousAnswer
    ){

        parts.push(

            "",

            "ПРЕДЫДУЩИЙ ОТВЕТ JESSICA:",

            previousAnswer

        );

    }


    /*
     * =====================================================
     * USER CORRECTION
     * =====================================================
     */


    parts.push(

        "",

        "ИСПРАВЛЕНИЕ ПОЛЬЗОВАТЕЛЯ:",

        correction

    );


    /*
     * =====================================================
     * CORRECTED ANSWER
     * =====================================================
     */


    if(
        correctedAnswer
    ){

        parts.push(

            "",

            "ПРАВИЛЬНЫЙ ОТВЕТ:",

            correctedAnswer

        );

    }


    /*
     * =====================================================
     * ANALYSIS INSTRUCTION
     * =====================================================
     */


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
     * NORMALIZE INPUT
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


    /*
     * =====================================================
     * INPUT VALIDATION
     * =====================================================
     */


    if(
        !cleanTask
    ){

        throw new Error(
            "Learning Analyzer: исходная задача не указана"
        );

    }


    if(
        !cleanCorrection
    ){

        throw new Error(
            "Learning Analyzer: исправление пользователя не указано"
        );

    }


    /*
     * =====================================================
     * USER PROMPT
     * =====================================================
     */


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


    /*
     * =====================================================
     * AI REQUEST
     * =====================================================
     */


    try {


        const response =

            await requestLearningAnalyzerAI({

                systemPrompt:
                    LEARNING_ANALYZER_SYSTEM_PROMPT,

                userPrompt,

                responseFormat:
                    LEARNING_ANALYZER_RESPONSE_FORMAT

            });


        /*
         * =================================================
         * CONTENT
         * =================================================
         */


        const content =

            response
                ?.choices
                ?.[0]
                ?.message
                ?.content;


        if(
            !content
        ){

            throw new Error(
                "Learning Analyzer: AI вернул пустой ответ"
            );

        }


        return content;


    }catch(error){


        /*
         * Technical AI retry уже был выполнен
         * внутри learningAnalyzerClient.
         *
         * Здесь ошибка только логируется
         * и передаётся вызывающему модулю.
         */


        console.error(

            "Jessica Learning Analyzer request failed:",

            {

                status:
                    error?.status || 0,

                message:
                    error?.message || String(error),

                code:
                    error?.code || null,

                type:
                    error?.type || null

            }

        );


        throw error;

    }

}
