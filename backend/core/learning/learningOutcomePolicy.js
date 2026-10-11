/*
 * =========================================================
 * JESSICA LEARNING OUTCOME POLICY v1
 * =========================================================
 *
 * Определяет:
 *
 * можно ли Execution использовать
 * как POSITIVE Experience Evidence.
 *
 *
 * Важно:
 *
 * Task completion
 *
 * НЕ равно
 *
 * Positive Learning Evidence.
 *
 *
 * Например:
 *
 * valid=true
 * outcomeType=no_verified_result
 *
 * означает:
 *
 * Jessica корректно решила задачу,
 * признав отсутствие подтверждённого результата.
 *
 * Но это НЕ является доказательством:
 *
 * "Skill успешно получил искомый результат".
 *
 *
 * Аналогично:
 *
 * degraded=true
 *
 * означает, что задача могла завершиться
 * успешно через fallback,
 * но качество проверки недостаточно
 * для полноценного reinforcement.
 *
 * =========================================================
 */


/*
 * =========================================================
 * CONSTANTS
 * =========================================================
 */


export const LEARNING_OUTCOME = {

    POSITIVE_RESULT:
        "POSITIVE_RESULT",

    NO_VERIFIED_RESULT:
        "NO_VERIFIED_RESULT",

    DEGRADED_VALIDATION:
        "DEGRADED_VALIDATION",

    VALIDATION_FAILED:
        "VALIDATION_FAILED",

    NEEDS_CLARIFICATION:
        "NEEDS_CLARIFICATION",

    EXECUTION_FAILED:
        "EXECUTION_FAILED",

    NOT_COMPLETED:
        "NOT_COMPLETED"

};


/*
 * =========================================================
 * HELPERS
 * =========================================================
 */


function isObject(
    value
) {

    return Boolean(

        value

        &&

        typeof value === "object"

        &&

        !Array.isArray(value)

    );

}


function normalizeText(
    value
) {

    return String(
        value || ""
    )
        .trim();

}


function normalizeUpper(
    value
) {

    return normalizeText(
        value
    )
        .toUpperCase();

}


function normalizeLower(
    value
) {

    return normalizeText(
        value
    )
        .toLowerCase();

}


function normalizeNumber(
    value
) {

    const number =
        Number(value);


    return Number.isFinite(number)

        ? number

        : 0;

}


/*
 * =========================================================
 * VALIDATION
 * =========================================================
 */


function resolveValidation(
    trace
) {

    if(
        isObject(
            trace?.result?.validation
        )
    ){

        return trace.result.validation;

    }


    if(
        isObject(
            trace?.validation
        )
    ){

        return trace.validation;

    }


    return null;

}


/*
 * =========================================================
 * OUTCOME TYPE
 * =========================================================
 */


function resolveOutcomeType(
    trace,
    validation
) {

    return normalizeLower(

        validation?.outcomeType

        ||

        trace?.result?.outcomeType

        ||

        trace?.outcomeType

    );

}


/*
 * =========================================================
 * STATUS
 * =========================================================
 */


function resolveStatus(
    trace
) {

    return normalizeUpper(

        trace?.result?.status

        ||

        trace?.status

    );

}


/*
 * =========================================================
 * DEGRADED
 * =========================================================
 */


function resolveDegraded(
    trace,
    validation
) {

    return (

        validation?.degraded === true

        ||

        trace?.result?.degraded === true

        ||

        trace?.degraded === true

    );

}


/*
 * =========================================================
 * COMPLETION
 * =========================================================
 */


function resolveCompleted(
    trace,
    status
) {

    const completed =

        normalizeNumber(

            trace?.statistics?.completed

            ??

            trace?.stats?.completed

        );


    if(
        completed > 0
    ){

        return true;

    }


    if(
        status === "COMPLETED"
    ){

        return true;

    }


    if(
        trace?.result?.success === true
    ){

        return true;

    }


    if(
        trace?.success === true
    ){

        return true;

    }


    return false;

}


/*
 * =========================================================
 * CLASSIFY
 * =========================================================
 */


export function classifyLearningOutcome(
    trace
) {

    if(
        !isObject(
            trace
        )
    ){

        return {

            positive:
                false,

            completed:
                false,

            outcome:
                LEARNING_OUTCOME.NOT_COMPLETED,

            outcomeType:
                null,

            degraded:
                false,

            validationMode:
                null,

            reason:
                "Execution Trace отсутствует"

        };

    }


    const validation =

        resolveValidation(
            trace
        );


    const outcomeType =

        resolveOutcomeType(

            trace,

            validation

        );


    const status =

        resolveStatus(
            trace
        );


    const degraded =

        resolveDegraded(

            trace,

            validation

        );


    const completed =

        resolveCompleted(

            trace,

            status

        );


    const validationMode =

        normalizeText(
            validation?.validationMode
        )

        ||

        null;


    /*
     * =====================================================
     * 1. NO VERIFIED RESULT
     * =====================================================
     */


    if(
        outcomeType ===
        "no_verified_result"

        ||

        status ===
        "NO_VERIFIED_RESULT"
    ){

        return {

            positive:
                false,

            completed,

            outcome:
                LEARNING_OUTCOME.NO_VERIFIED_RESULT,

            outcomeType:
                "no_verified_result",

            degraded,

            validationMode,

            reason:

                "Execution корректно завершён без подтверждённого положительного результата"

        };

    }


    /*
     * =====================================================
     * 2. NEEDS CLARIFICATION
     * =====================================================
     */


    if(
        status ===
        "NEEDS_CLARIFICATION"

        ||

        validation?.needsClarification === true
    ){

        return {

            positive:
                false,

            completed,

            outcome:
                LEARNING_OUTCOME.NEEDS_CLARIFICATION,

            outcomeType:

                outcomeType

                ||

                null,

            degraded,

            validationMode,

            reason:
                "Execution требует уточнения"

        };

    }


    /*
     * =====================================================
     * 3. EXECUTION FAILED
     * =====================================================
     */


    if(
        status === "FAILED"

        ||

        trace?.result?.success === false

        ||

        trace?.success === false
    ){

        return {

            positive:
                false,

            completed,

            outcome:
                LEARNING_OUTCOME.EXECUTION_FAILED,

            outcomeType:

                outcomeType

                ||

                null,

            degraded,

            validationMode,

            reason:
                "Execution завершился ошибкой"

        };

    }


    /*
     * =====================================================
     * 4. VALIDATION FAILED
     * =====================================================
     */


    if(
        validation
        &&
        validation.valid === false
    ){

        return {

            positive:
                false,

            completed,

            outcome:
                LEARNING_OUTCOME.VALIDATION_FAILED,

            outcomeType:

                outcomeType

                ||

                null,

            degraded,

            validationMode,

            reason:
                "Result Validation не подтверждает результат"

        };

    }


    /*
     * =====================================================
     * 5. DEGRADED VALIDATION
     * =====================================================
     *
     * Task может считаться завершённой,
     * но такой Execution не должен
     * усиливать Experience как обычный
     * verified success.
     * =====================================================
     */


    if(
        degraded
    ){

        return {

            positive:
                false,

            completed,

            outcome:
                LEARNING_OUTCOME.DEGRADED_VALIDATION,

            outcomeType:

                outcomeType

                ||

                "result",

            degraded:
                true,

            validationMode,

            reason:
                "Execution прошёл через degraded validation path"

        };

    }


    /*
     * =====================================================
     * 6. NOT COMPLETED
     * =====================================================
     */


    if(
        !completed
    ){

        return {

            positive:
                false,

            completed:
                false,

            outcome:
                LEARNING_OUTCOME.NOT_COMPLETED,

            outcomeType:

                outcomeType

                ||

                null,

            degraded:
                false,

            validationMode,

            reason:
                "Execution не подтверждён как завершённый"

        };

    }


    /*
     * =====================================================
     * 7. POSITIVE RESULT
     * =====================================================
     */


    return {

        positive:
            true,

        completed:
            true,

        outcome:
            LEARNING_OUTCOME.POSITIVE_RESULT,

        outcomeType:

            outcomeType

            ||

            "result",

        degraded:
            false,

        validationMode,

        reason:
            "Execution содержит положительно подтверждённый результат"

    };

}


/*
 * =========================================================
 * POSITIVE
 * =========================================================
 */


export function isPositiveLearningTrace(
    trace
) {

    return (

        classifyLearningOutcome(
            trace
        )
        .positive === true

    );

}
