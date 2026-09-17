/*
 * =========================================================
 * JESSICA DIRECT ANSWER
 * =========================================================
 *
 * Обрабатывает инструменты,
 * которые уже возвращают готовый
 * пользовательский ответ.
 *
 *
 * Для таких инструментов
 * AI Answer Composer не нужен.
 *
 *
 * НЕ отвечает за:
 *
 * - AI request;
 * - prompt;
 * - factual context;
 * - validation;
 * - execution;
 * - semantic outcome.
 *
 * =========================================================
 */


/*
 * =========================================================
 * DIRECT ANSWER TOOLS
 * =========================================================
 */


const DIRECT_ANSWER_TOOLS =
    new Set([

        "current_time"

    ]);


/*
 * =========================================================
 * NORMALIZE TEXT
 * =========================================================
 */


function normalizeAnswerText(
    value
) {

    return typeof value === "string"
        ? value.trim()
        : "";

}


/*
 * =========================================================
 * SINGLE RESULT
 * =========================================================
 */


function getSingleToolResult(
    taskRunResult
) {

    if (
        !taskRunResult ||
        !Array.isArray(
            taskRunResult.results
        )
    ) {

        return null;

    }


    if (
        taskRunResult.results.length !== 1
    ) {

        return null;

    }


    return taskRunResult.results[0] || null;

}


/*
 * =========================================================
 * DIRECT ANSWER TOOL
 * =========================================================
 */


function isDirectAnswerTool(
    tool
) {

    return DIRECT_ANSWER_TOOLS.has(
        tool
    );

}


/*
 * =========================================================
 * PUBLIC
 * =========================================================
 */


export function getDirectAnswer(
    taskRunResult
) {

    const result =
        getSingleToolResult(
            taskRunResult
        );


    if (!result) {

        return null;

    }


    if (
        result.success !== true
    ) {

        return null;

    }


    if (
        !isDirectAnswerTool(
            result.tool
        )
    ) {

        return null;

    }


    const text =
        normalizeAnswerText(
            result.text
        );


    if (!text) {

        return null;

    }


    return {

        success:
            true,

        text,

        source:
            "tool",

        tool:
            result.tool

    };

}


/*
 * =========================================================
 * TOOL INFO
 * =========================================================
 */


export function isDirectAnswerToolName(
    tool
) {

    return isDirectAnswerTool(
        tool
    );

}
