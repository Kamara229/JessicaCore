/*
 * =========================================================
 * JESSICA AI RESULT VALIDATOR
 * =========================================================
 *
 * Центральный координатор AI-проверки результата.
 *
 *
 * Flow:
 *
 * task / plan / runResult / answer
 *        ↓
 * aiResultPrompt
 *        ↓
 * aiResultRequest
 *        ↓
 * AI
 *        ↓
 * aiResultParser
 *        ↓
 * normalized validation result
 *
 *
 * Этот файл НЕ:
 *
 * - строит prompt;
 * - сокращает TaskRunner payload;
 * - вызывает validatorChat напрямую;
 * - содержит retry-логику;
 * - парсит JSON;
 * - нормализует outcomeType.
 *
 * =========================================================
 */


import {
    buildAIResultValidatorMessages
} from "./aiResult/aiResultPrompt.js";

import {
    requestAIResultValidation
} from "./aiResult/aiResultRequest.js";

import {
    parseAIResultValidatorResponse
} from "./aiResult/aiResultParser.js";


/*
 * =========================================================
 * ERROR RESULT
 * =========================================================
 */


function buildErrorResult(
    error
) {

    return {

        success:
            false,


        unavailable:
            true,


        status:
            error?.status || 0,


        reason:
            error?.message ||
            "Ошибка AI Validator"

    };

}


/*
 * =========================================================
 * PUBLIC
 * =========================================================
 */


export async function validateWithAI(

    task,

    plan,

    taskRunResult,

    answerResult

) {

    try {


        /*
         * =================================================
         * 1. BUILD MESSAGES
         * =================================================
         */


        const messages =
            buildAIResultValidatorMessages(

                task,

                plan,

                taskRunResult,

                answerResult

            );


        /*
         * =================================================
         * 2. AI REQUEST
         * =================================================
         */


        const response =
            await requestAIResultValidation(
                messages
            );


        /*
         * =================================================
         * 3. PARSE + NORMALIZE
         * =================================================
         */


        return parseAIResultValidatorResponse(
            response
        );


    } catch (error) {


        console.error(
            "AI Validator error:",
            error
        );


        return buildErrorResult(
            error
        );

    }

}
