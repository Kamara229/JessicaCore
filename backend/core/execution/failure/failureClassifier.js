/*
 * =========================================================
 * JESSICA EXECUTION
 * FAILURE CLASSIFIER v1
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
 * DETECT CATEGORY
 * =========================================================
 */


export function classifyFailure(

    failure

) {


    const type =

        String(

            failure?.failureType ||

            ""

        )
        .toLowerCase();









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
