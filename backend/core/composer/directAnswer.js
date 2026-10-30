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

        "current_time",

        "date_diff",

        "invalid_calendar_date",

        "release_fact"

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


function getDirectToolResult(
    taskRunResult
) {

    if (
        !taskRunResult ||
        !Array.isArray(
            taskRunResult.results
        ) ||
        taskRunResult.results.length === 0
    ) {

        return null;

    }


    /*
     * Supporting steps may precede a deterministic
     * final tool:
     *
     * web_search → web_fetch → release_fact
     *
     * The final direct tool may answer without Composer
     * only when all previous steps also succeeded.
     */


    const results =
        taskRunResult.results;


    if(
        results.some(
            item =>
                item?.success !== true
        )
    ){

        return null;

    }


    const result =
        results[
            results.length - 1
        ];


    return result || null;

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
        getDirectToolResult(
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
