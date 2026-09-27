/*
 * =========================================================
 * JESSICA EXECUTION
 * FAILURE CLASSIFIER v3
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







function getFailureType(

    failure

) {


    return String(

        failure?.failureType ||

        ""

    )
    .toLowerCase()
    .trim();

}









export function classifyFailure(

    failure

) {


    if(

        !failure ||

        typeof failure !== "object"

    ){

        return "execution";

    }









    /*
     * Если категория уже определена
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
     * USER INPUT
     */


    if(

        failure.needsClarification === true

    ){

        return "clarification";

    }









    /*
     * RESULT NOT VERIFIED
     */


    if(

        failure.noVerifiedResult === true

    ){

        return "no_verified";

    }









    /*
     * VALIDATION
     */


    if(

        type.includes("validation")

        ||

        type.includes("verify")

        ||

        type.includes("check")

    ){

        return "validation";

    }









    /*
     * TOOL
     */


    if(

        type.includes("tool")

        ||

        type.includes("api")

        ||

        type.includes("connector")

    ){

        return "tool";

    }









    /*
     * TEMPORARY
     */


    if(

        type.includes("timeout")

        ||

        type.includes("network")

        ||

        type.includes("temporary")

        ||

        type.includes("rate")

        ||

        type.includes("service")

        ||

        type.includes("unavailable")

        ||

        type.includes("connection")

        ||

        type.includes("busy")

    ){

        return "temporary";

    }









    /*
     * PLANNER
     */


    if(

        type.includes("planner")

        ||

        type.includes("strategy")

        ||

        type.includes("plan")

        ||

        type.includes("route")

    ){

        return "planner";

    }









    /*
     * DATA
     */


    if(

        type.includes("data")

        ||

        type.includes("schema")

        ||

        type.includes("argument")

        ||

        type.includes("parameter")

        ||

        type.includes("format")

    ){

        return "data";

    }









    return "execution";


}
