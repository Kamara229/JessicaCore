/*
 * =========================================================
 * JESSICA EXECUTION
 * FAILURE CLASSIFIER v4
 * =========================================================
 *
 * Определение категории Execution Failure.
 *
 *
 * Categories:
 *
 * validation
 * tool
 * temporary
 * planner
 * data
 * clarification
 * no_verified
 * presentation
 * execution
 *
 *
 * Presentation:
 *
 * ответ уже должен быть сформирован
 * на основании результатов Execution,
 * но presentation layer не смог
 * собрать пользовательский текст.
 *
 *
 * Presentation failure НЕ является
 * причиной менять Execution Route.
 *
 * =========================================================
 */


function getFailureType(
    failure
) {

    return String(

        failure?.failureType

        ||

        ""

    )
        .toLowerCase()
        .trim();

}


/*
 * =========================================================
 * CLASSIFY
 * =========================================================
 */


export function classifyFailure(
    failure
) {

    if(
        !failure
        ||
        typeof failure !== "object"
    ){

        return "execution";

    }


    /*
     * =====================================================
     * EXPLICIT CATEGORY
     * =====================================================
     */


    if(
        failure.category
    ){

        return String(
            failure.category
        )
            .toLowerCase()
            .trim();

    }


    const type =

        getFailureType(
            failure
        );


    /*
     * =====================================================
     * USER INPUT
     * =====================================================
     */


    if(
        failure.needsClarification
        ===
        true
    ){

        return "clarification";

    }


    /*
     * =====================================================
     * NO VERIFIED RESULT
     * =====================================================
     */


    if(
        failure.noVerifiedResult
        ===
        true
    ){

        return "no_verified";

    }


    /*
     * =====================================================
     * PRESENTATION
     * =====================================================
     *
     * Проверяем раньше generic execution.
     *
     * =====================================================
     */


    if(
        type.includes(
            "composer"
        )

        ||

        type.includes(
            "presentation"
        )

        ||

        type.includes(
            "answer-generation"
        )

        ||

        type.includes(
            "answer-composition"
        )
    ){

        return "presentation";

    }


    /*
     * =====================================================
     * VALIDATION
     * =====================================================
     */


    if(
        type.includes(
            "validation"
        )

        ||

        type.includes(
            "verify"
        )

        ||

        type.includes(
            "check"
        )
    ){

        return "validation";

    }


    /*
     * =====================================================
     * TOOL
     * =====================================================
     */


    if(
        type.includes(
            "tool"
        )

        ||

        type.includes(
            "api"
        )

        ||

        type.includes(
            "connector"
        )
    ){

        return "tool";

    }


    /*
     * =====================================================
     * TEMPORARY
     * =====================================================
     */


    if(
        type.includes(
            "timeout"
        )

        ||

        type.includes(
            "network"
        )

        ||

        type.includes(
            "temporary"
        )

        ||

        type.includes(
            "rate"
        )

        ||

        type.includes(
            "service"
        )

        ||

        type.includes(
            "unavailable"
        )

        ||

        type.includes(
            "connection"
        )

        ||

        type.includes(
            "busy"
        )
    ){

        return "temporary";

    }


    /*
     * =====================================================
     * PLANNER
     * =====================================================
     */


    if(
        type.includes(
            "planner"
        )

        ||

        type.includes(
            "strategy"
        )

        ||

        type.includes(
            "plan"
        )

        ||

        type.includes(
            "route"
        )
    ){

        return "planner";

    }


    /*
     * =====================================================
     * DATA
     * =====================================================
     */


    if(
        type.includes(
            "data"
        )

        ||

        type.includes(
            "schema"
        )

        ||

        type.includes(
            "argument"
        )

        ||

        type.includes(
            "parameter"
        )

        ||

        type.includes(
            "format"
        )
    ){

        return "data";

    }


    return "execution";

}
