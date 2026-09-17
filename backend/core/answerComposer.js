/*
 * =========================================================
 * JESSICA ANSWER COMPOSER
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

import {
    isComposerAvailable,
    requestComposerAnswer
} from "./composer/composerRequest.js";


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
    taskRunResult
) {

    try {

        return {

            success:
                true,

            input:
                buildComposerContext(

                    task,

                    plan,

                    taskRunResult

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

    taskRunResult

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


    /*
     * =====================================================
     * 2. AI AVAILABILITY
     * =====================================================
     */


    if (
        !isComposerAvailable()
    ) {

        return buildComposerFailure(
            "Groq Answer Composer не настроен"
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

            taskRunResult

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


    return await generateComposerAnswer(
        contextResult.input
    );

}
