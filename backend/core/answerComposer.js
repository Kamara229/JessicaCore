/*
 * =========================================================
 * JESSICA ANSWER COMPOSER v3
 * =========================================================
 *
 * Центральный координатор формирования
 * пользовательского ответа.
 *
 *
 * Flow:
 *
 * TaskRunResult
 *      ↓
 * Direct Answer?
 *      │
 *      ├── yes → return
 *      │
 *      └── no
 *           ↓
 * Composer Context
 *           ↓
 * Composer Request
 *           ↓
 * Response Check
 *           ↓
 * Final Answer
 *
 *
 * Специализированная логика вынесена в:
 *
 * composer/
 *
 * ├── composerContext.js
 * ├── composerPrompt.js
 * ├── composerRequest.js
 * └── directAnswer.js
 *
 *
 * Этот файл НЕ:
 *
 * - создаёт AI client;
 * - хранит prompt;
 * - вызывает tools;
 * - строит factual context вручную;
 * - содержит direct-tool правила;
 * - валидирует факты ответа;
 * - определяет semantic outcome.
 *
 * =========================================================
 */


import {
    buildComposerContext
} from "./composer/composerContext.js";

import {
    getDirectAnswer
} from "./composer/directAnswer.js";
import { getDocumentReceipt } from "./composer/documentReceipt.js";

import {
    isComposerAvailable,
    requestComposerAnswer
} from "./composer/composerRequest.js";


import {
    buildNativeComposerFallback
} from "./composer/nativeComposerFallback.js";


/*
 * =========================================================
 * FAILURE RESULT
 * =========================================================
 */


function buildComposerFailure(
    text,
    options = {}
) {

    return {

        success:
            false,

        text:
            text ||
            "Не удалось сформировать итоговый ответ",

        status:
            options?.status || 0,

        retryable:
            options?.retryable === true

    };

}


/*
 * =========================================================
 * EXTRACT AI MESSAGE
 * =========================================================
 */


function extractComposerMessage(
    response
) {

    return response
        ?.choices
        ?.[0]
        ?.message ||
        null;

}


/*
 * =========================================================
 * TOOL CALL GUARD
 * =========================================================
 *
 * Composer не должен выполнять действия.
 *
 * Даже если модель неожиданно попытается
 * сформировать tool call, такой ответ
 * блокируется.
 *
 * =========================================================
 */


function hasToolCalls(
    message
) {

    return (
        Array.isArray(
            message?.tool_calls
        ) &&
        message.tool_calls.length > 0
    );

}


/*
 * =========================================================
 * EXTRACT ANSWER TEXT
 * =========================================================
 */


function extractAnswerText(
    message
) {

    return typeof message?.content === "string"
        ? message.content.trim()
        : "";

}


/*
 * =========================================================
 * BUILD CONTEXT
 * =========================================================
 */


function prepareComposerContext(
    task,
    plan,
    taskRunResult,
    repairContext = null
) {

    try {

        return {

            success:
                true,

            input:
                buildComposerContext(

                    task,

                    plan,

                    taskRunResult,

                    repairContext

                )

        };

    } catch (error) {

        console.error(
            "Answer Composer context error:",
            error
        );


        return {

            success:
                false,

            result:
                buildComposerFailure(
                    "Не удалось подготовить данные для итогового ответа"
                )

        };

    }

}


/*
 * =========================================================
 * REQUEST AI ANSWER
 * =========================================================
 */


async function generateComposerAnswer(
    input
) {

    try {

        const response =
            await requestComposerAnswer(
                input
            );


        const message =
            extractComposerMessage(
                response
            );


        /*
         * =================================================
         * TOOL CALL PROTECTION
         * =================================================
         */


        if (
            hasToolCalls(
                message
            )
        ) {

            console.error(
                "Answer Composer attempted tool call:",
                message.tool_calls
            );


            return buildComposerFailure(
                "Answer Composer попытался выполнить недопустимое действие"
            );

        }


        /*
         * =================================================
         * ANSWER TEXT
         * =================================================
         */


        const answer =
            extractAnswerText(
                message
            );


        if (!answer) {

            return buildComposerFailure(
                "Answer Composer вернул пустой ответ"
            );

        }


        /*
         * =================================================
         * SUCCESS
         * =================================================
         */


        return {

            success:
                true,

            text:
                answer,

            source:
                "groq"

        };

    } catch (error) {

        console.error(
            "Answer Composer final error:",
            error
        );


        return buildComposerFailure(

            "Не удалось сформировать итоговый ответ",

            {

                status:
                    error?.status || 0,

                retryable:
                    error?.status === 429

            }

        );

    }

}


/*
 * =========================================================
 * PUBLIC
 * =========================================================
 */


export async function composeAnswer(

    task,

    plan,

    taskRunResult,

    options = {}

) {

    /*
     * =====================================================
     * 1. DIRECT TOOL ANSWER
     * =====================================================
     */


    const directAnswer =
        getDirectAnswer(
            taskRunResult
        );


    if (directAnswer) {

        return directAnswer;

    }


    // Source capture is a verified receipt, not a synthesized answer.
    // Avoid Groq composition for a fetch-only prerequisite.
    const receipt = getDocumentReceipt(task, plan, taskRunResult);
    if (receipt) return receipt;
    if (plan?.intent === "native_document_capture") {
        return buildComposerFailure("Документ не подтверждён, неполон либо адрес источника не совпадает.");
    }

    /*
     * =====================================================
     * 2. AI AVAILABILITY
     * =====================================================
     */


    if (
        !isComposerAvailable()
    ) {

        const fallback =

            buildNativeComposerFallback(

                task,

                taskRunResult

            );


        if(fallback){

            console.log(

                "Jessica Answer Composer native fallback:",

                {

                    reason:
                        "composer-unavailable",

                    source:
                        fallback.source

                }

            );


            return fallback;

        }


        return buildComposerFailure(
            "Answer Composer недоступен"
        );

    }


    /*
     * =====================================================
     * 3. BUILD FACTUAL CONTEXT
     * =====================================================
     */


    const contextResult =
        prepareComposerContext(

            task,

            plan,

            taskRunResult,

            options?.repairContext || null

        );


    if (
        !contextResult.success
    ) {

        return contextResult.result;

    }


    /*
     * =====================================================
     * 4. GENERATE FINAL ANSWER
     * =====================================================
     */


    const generated =

        await generateComposerAnswer(
            contextResult.input
        );


    if(
        generated?.success === true
    ){

        return generated;

    }


    /*
     * Technical AI failure не уничтожает
     * уже полученные Tool Results.
     */


    const fallback =

        buildNativeComposerFallback(

            task,

            taskRunResult

        );


    if(fallback){

        console.log(

            "Jessica Answer Composer native fallback:",

            {

                reason:
                    "composer-technical-failure",

                status:
                    generated?.status || 0,

                source:
                    fallback.source

            }

        );


        return fallback;

    }


    return generated;

}
