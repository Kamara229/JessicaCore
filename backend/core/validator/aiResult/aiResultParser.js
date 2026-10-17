/*
 * =========================================================
 * JESSICA AI RESULT PARSER v2
 * =========================================================
 *
 * Разбирает ответ AI Result Validator
 * и сохраняет semantic failureScope.
 *
 * =========================================================
 */


export const AI_RESULT_OUTCOME = {

    RESULT:
        "result",

    NO_VERIFIED_RESULT:
        "no_verified_result"

};


export const AI_RESULT_FAILURE_SCOPE = {

    PRESENTATION:
        "presentation",

    EXECUTION:
        "execution",

    EVIDENCE:
        "evidence",

    CLARIFICATION:
        "clarification"

};


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


    if(
        firstBrace !== -1
        &&
        lastBrace > firstBrace
    ){

        value =
            value.slice(
                firstBrace,
                lastBrace + 1
            );

    }


    return value;

}


function normalizeOutcomeType(
    value
) {

    if(
        value ===
        AI_RESULT_OUTCOME.NO_VERIFIED_RESULT
    ){

        return AI_RESULT_OUTCOME.NO_VERIFIED_RESULT;

    }


    return AI_RESULT_OUTCOME.RESULT;

}


function normalizeReason(
    value
) {

    return typeof value === "string"

        ? value.trim()

        : "";

}


function normalizeFailureScope(
    validation
) {

    if(
        validation?.valid === true
    ){

        return null;

    }


    if(
        validation?.needsClarification === true
    ){

        return AI_RESULT_FAILURE_SCOPE.CLARIFICATION;

    }


    const value =
        String(
            validation?.failureScope || ""
        )
            .trim()
            .toLowerCase();


    if(
        Object.values(
            AI_RESULT_FAILURE_SCOPE
        )
            .includes(
                value
            )
    ){

        return value;

    }


    /*
     * Conservative fallback:
     *
     * если Validator не указал scope,
     * нельзя автоматически считать ошибку
     * presentation и скрывать проблему
     * Execution.
     */

    return AI_RESULT_FAILURE_SCOPE.EXECUTION;

}


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


        failureScope:
            normalizeFailureScope(
                validation
            ),


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


function buildEmptyResponseResult()
{

    return {

        success:
            false,

        unavailable:
            false,

        reason:
            "AI Validator вернул пустой ответ"

    };

}


function buildInvalidJsonResult()
{

    return {

        success:
            false,

        unavailable:
            false,

        reason:
            "AI Validator вернул некорректный JSON"

    };

}


function parseContent(
    raw
) {

    if(
        typeof raw !== "string"
        ||
        !raw.trim()
    ){

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


    if(
        !validation
        ||
        typeof validation !== "object"
        ||
        Array.isArray(
            validation
        )
    ){

        return buildInvalidJsonResult();

    }


    return normalizeValidation(
        validation
    );

}


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
