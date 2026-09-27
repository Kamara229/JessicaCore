/*
 * =========================================================
 * JESSICA RUN FAILURE POLICY v5
 * =========================================================
 *
 * Классификатор Execution Failure.
 *
 *
 * Flow:
 *
 * Execution Output
 *        ↓
 * Normalize
 *        ↓
 * Classify
 *        ↓
 * Normalized Failure
 *
 *
 * Ответственность:
 *
 * - определить тип ошибки;
 * - определить категорию;
 * - подготовить Failure объект.
 *
 *
 * НЕ:
 *
 * - решает Retry;
 * - решает Replan;
 * - вызывает Planner;
 * - вызывает Tools;
 * - создаёт Terminal Result.
 *
 * =========================================================
 */







/*
 * =========================================================
 * FAILURE TYPES
 * =========================================================
 */


export const FAILURE_TYPE = {


    TOOL_ERROR:
        "tool-error",


    RUNNER_ERROR:
        "runner-error",


    COMPOSER_ERROR:
        "composer-error",


    VALIDATOR_ERROR:
        "validation-error",


    INVALID_RESULT:
        "invalid-result",


    MISSING_DATA:
        "missing-data",


    USER_REQUIRED:
        "user-required",


    TEMPORARY_ERROR:
        "temporary-error",


    TIMEOUT:
        "timeout-error",


    NETWORK:
        "network-error",


    NO_VERIFIED_RESULT:
        "no-verified-result",


    UNKNOWN:
        "unknown"


};









/*
 * =========================================================
 * FAILURE CATEGORIES
 * =========================================================
 */


export const FAILURE_CATEGORY = {


    TEMPORARY:
        "temporary",


    TOOL:
        "tool",


    VALIDATION:
        "validation",


    DATA:
        "data",


    USER:
        "user",


    EXECUTION:
        "execution"



};









/*
 * =========================================================
 * TEXT
 * =========================================================
 */


function normalizeText(
    value
) {


    return String(
        value || ""
    )
    .trim();


}









/*
 * =========================================================
 * CATEGORY DETECTION
 * =========================================================
 */


function detectCategory(

    failureType

) {


    switch(
        failureType
    ){


        case FAILURE_TYPE.TIMEOUT:


        case FAILURE_TYPE.NETWORK:


        case FAILURE_TYPE.TEMPORARY_ERROR:


            return FAILURE_CATEGORY.TEMPORARY;





        case FAILURE_TYPE.TOOL_ERROR:


            return FAILURE_CATEGORY.TOOL;





        case FAILURE_TYPE.VALIDATOR_ERROR:


        case FAILURE_TYPE.INVALID_RESULT:


            return FAILURE_CATEGORY.VALIDATION;





        case FAILURE_TYPE.MISSING_DATA:


            return FAILURE_CATEGORY.DATA;





        case FAILURE_TYPE.USER_REQUIRED:


            return FAILURE_CATEGORY.USER;





        default:


            return FAILURE_CATEGORY.EXECUTION;


    }


}









/*
 * =========================================================
 * BUILD FAILURE
 * =========================================================
 */


function buildFailure({

    stage,

    failureType,

    reason,

    original = null,

    extra = {}

}) {


    return {


        failed:
            true,


        stage:

            stage ||
            "execution",



        failureType:



            failureType ||
            FAILURE_TYPE.UNKNOWN,



        category:



            detectCategory(
                failureType
            ),



        reason:



            reason ||
            "Ошибка выполнения",



        needsClarification:

            failureType ===
            FAILURE_TYPE.USER_REQUIRED,



        noVerifiedResult:

            failureType ===
            FAILURE_TYPE.NO_VERIFIED_RESULT,



        original,



        ...extra


    };


}









/*
 * =========================================================
 * DETECT TYPE
 * =========================================================
 */


function detectFailureType(

    result,

    reason

) {


    const text =

        reason
            .toLowerCase();








    if(

        result?.needsClarification === true

    ){

        return FAILURE_TYPE.USER_REQUIRED;

    }








    if(

        result?.stage === "tool"

    ){

        return FAILURE_TYPE.TOOL_ERROR;

    }








    if(

        result?.stage === "runner"

    ){

        return FAILURE_TYPE.RUNNER_ERROR;

    }








    if(

        result?.stage === "composer"

    ){

        return FAILURE_TYPE.COMPOSER_ERROR;

    }








    if(

        result?.stage === "validator"

    ){

        return FAILURE_TYPE.VALIDATOR_ERROR;

    }








    if(

        text.includes("timeout")

        ||

        text.includes("timed out")

    ){

        return FAILURE_TYPE.TIMEOUT;

    }








    if(

        text.includes("network")

        ||

        text.includes("connection")

    ){

        return FAILURE_TYPE.NETWORK;

    }








    if(

        text.includes("temporary")

        ||

        text.includes("temporarily")

    ){

        return FAILURE_TYPE.TEMPORARY_ERROR;

    }








    if(

        text.includes("missing")

        ||

        text.includes("required")

        ||

        text.includes("не указан")

        ||

        text.includes("отсутствует")

    ){

        return FAILURE_TYPE.MISSING_DATA;

    }








    if(

        text.includes("validation")

        ||

        text.includes("провер")

        ||

        text.includes("не прошёл")

    ){

        return FAILURE_TYPE.VALIDATOR_ERROR;

    }








    if(

        result?.noVerifiedResult === true

    ){

        return FAILURE_TYPE.NO_VERIFIED_RESULT;

    }








    return FAILURE_TYPE.UNKNOWN;


}









/*
 * =========================================================
 * ANALYZE EXECUTION RESULT
 * =========================================================
 */


export function analyzeRunFailure(

    executionResult

) {



    /*
     * SUCCESS
     */


    if(

        executionResult?.success === true

    ){


        return {


            failed:false,


            failureType:null,


            category:null,


            stage:null,


            reason:""


        };


    }









    const reason =

        normalizeText(

            executionResult?.failure?.reason

            ||

            executionResult?.reason

            ||

            executionResult?.text

            ||

            executionResult?.error

        );








    const detectedType =

        executionResult?.failureType

        ||

        executionResult?.failure?.failureType

        ||

        detectFailureType(

            executionResult,

            reason

        );








    return buildFailure({

        stage:

            executionResult?.stage

            ||

            executionResult?.failure?.stage

            ||

            "execution",



        failureType:

            detectedType,



        reason:

            reason ||

            "Не удалось выполнить операцию",



        original:

            executionResult,


        extra:

        {

            validation:

                executionResult?.validation

                ||

                executionResult?.failure?.validation

                ||

                null


        }


    });


}









/*
 * =========================================================
 * REPLANNER FEEDBACK
 * =========================================================
 *
 * Только перенос данных.
 *
 * Не решение.
 *
 * =========================================================
 */


export function buildRunFailureFeedback(

    failure

) {


    if(
        !failure
    ){

        return null;

    }



    return {


        failureType:

            failure.failureType,



        category:

            failure.category,



        stage:

            failure.stage,



        reason:

            failure.reason



    };


}
