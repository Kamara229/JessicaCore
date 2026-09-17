/*
 * =========================================================
 * JESSICA AI RESULT PARSER
 * =========================================================
 *
 * Разбирает ответ AI Result Validator.
 *
 *
 * Отвечает за:
 *
 * - извлечение content из AI response;
 * - очистку JSON;
 * - JSON.parse;
 * - нормализацию outcomeType;
 * - формирование единого validation result.
 *
 *
 * НЕ отвечает за:
 *
 * - построение prompt;
 * - вызов AI;
 * - retry;
 * - выполнение инструментов;
 * - semantic validation самостоятельно.
 *
 * =========================================================
 */


/*
 * =========================================================
 * OUTCOME TYPES
 * =========================================================
 */


export const AI_RESULT_OUTCOME = {

    RESULT:
        "result",

    NO_VERIFIED_RESULT:
        "no_verified_result"

};


/*
 * =========================================================
 * CLEAN JSON TEXT
 * =========================================================
 */


function cleanJsonText(
    text
) {

    let value =
        String(
            text || ""
        )
            .replace(
                /```json/gi,
                ""
            )
            .replace(
                /```/g,
                ""
            )
            .trim();


    const firstBrace =
        value.indexOf("{");


    const lastBrace =
        value.lastIndexOf("}");


    if (
        firstBrace !== -1 &&
        lastBrace > firstBrace
    ) {

        value =
            value.slice(
                firstBrace,
                lastBrace + 1
            );

    }


    return value;

}


/*
 * =========================================================
 * NORMALIZE OUTCOME TYPE
 * =========================================================
 */


function normalizeOutcomeType(
    value
) {

    if (
        value ===
        AI_RESULT_OUTCOME.NO_VERIFIED_RESULT
    ) {

        return AI_RESULT_OUTCOME.NO_VERIFIED_RESULT;

    }


    return AI_RESULT_OUTCOME.RESULT;

}


/*
 * =========================================================
 * NORMALIZE REASON
 * =========================================================
 */


function normalizeReason(
    value
) {

    return typeof value === "string"
        ? value.trim()
        : "";

}


/*
 * =========================================================
 * NORMALIZE VALIDATION
 * =========================================================
 */


function normalizeValidation(
    validation
) {

    return {

        success:
            true,


        valid:
            validation?.valid === true,


        shouldRetry:
            validation?.shouldRetry === true,


        needsClarification:
            validation?.needsClarification === true,


        outcomeType:
            normalizeOutcomeType(
                validation?.outcomeType
            ),


        reason:
            normalizeReason(
                validation?.reason
            )

    };

}


/*
 * =========================================================
 * EMPTY RESPONSE
 * =========================================================
 */


function buildEmptyResponseResult() {

    return {

        success:
            false,

        unavailable:
            false,

        reason:
            "AI Validator вернул пустой ответ"

    };

}


/*
 * =========================================================
 * INVALID JSON
 * =========================================================
 */


function buildInvalidJsonResult() {

    return {

        success:
            false,

        unavailable:
            false,

        reason:
            "AI Validator вернул некорректный JSON"

    };

}


/*
 * =========================================================
 * PARSE CONTENT
 * =========================================================
 */


function parseContent(
    raw
) {

    if (
        typeof raw !== "string" ||
        !raw.trim()
    ) {

        return buildEmptyResponseResult();

    }


    let validation;


    try {

        validation =
            JSON.parse(
                cleanJsonText(
                    raw
                )
            );

    } catch {

        return buildInvalidJsonResult();

    }


    if (
        !validation ||
        typeof validation !== "object" ||
        Array.isArray(
            validation
        )
    ) {

        return buildInvalidJsonResult();

    }


    return normalizeValidation(
        validation
    );

}


/*
 * =========================================================
 * PUBLIC
 * =========================================================
 */


export function parseAIResultValidatorResponse(
    response
) {

    const raw =
        response
            ?.choices
            ?.[0]
            ?.message
            ?.content;


    return parseContent(
        raw
    );

}
