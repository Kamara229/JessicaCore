/*
 * =========================================================
 * JESSICA EXECUTION
 * FAILURE CLASSIFIER v2
 * =========================================================
 *
 * Определение категории Execution Failure.
 *
 *
 * Flow:
 *
 * Normalized Failure
 *        ↓
 * Category
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
 * execution
 *
 *
 * НЕ:
 *
 * - делает Retry;
 * - делает Replan;
 * - создаёт Decision.
 *
 * =========================================================
 */







/*
 * =========================================================
 * SAFE TYPE
 * =========================================================
 */


function getFailureType(

    failure

) {


    return String(

        failure?.failureType ||

        ""

    )
    .toLowerCase();


}









/*
 * =========================================================
 * DETECT CATEGORY
 * =========================================================
 */


export function classifyFailure(

    failure

) {


    const type =

        getFailureType(

            failure

        );









    /*
     * =====================================================
     * CLARIFICATION
     * =====================================================
     */


    if (

        failure?.needsClarification === true

    ) {


        return "clarification";

    }









    /*
     * =====================================================
     * NO VERIFIED RESULT
     * =====================================================
     */


    if (

        failure?.noVerifiedResult === true

    ) {


        return "no_verified";

    }









    /*
     * =====================================================
     * VALIDATION
     * =====================================================
     */


    if (

        type.includes(
            "validation"
        )

    ) {


        return "validation";

    }









    /*
     * =====================================================
     * TOOL
     * =====================================================
     */


    if (

        type.includes(
            "tool"
        )

    ) {


        return "tool";

    }









    /*
     * =====================================================
     * TEMPORARY
     * =====================================================
     */


    if (

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

    ) {


        return "temporary";

    }









    /*
     * =====================================================
     * PLANNER
     * =====================================================
     */


    if (

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

    ) {


        return "planner";

    }









    /*
     * =====================================================
     * DATA
     * =====================================================
     */


    if (

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

    ) {


        return "data";

    }









    /*
     * =====================================================
     * DEFAULT
     * =====================================================
     */


    return "execution";


}
