/*
 * =========================================================
 * JESSICA EXECUTION RESULT v3
 * =========================================================
 *
 * Единый контракт результата Execution.
 *
 *
 * Создаёт:
 *
 * - COMPLETED
 * - FAILED
 * - NEEDS_CLARIFICATION
 * - NO_VERIFIED_RESULT
 *
 *
 * Используется:
 *
 * Execution Cycle
 * Terminal
 * Trace
 * Learning
 *
 *
 * НЕ:
 *
 * - анализирует ошибки;
 * - решает retry;
 * - делает replan;
 * - вызывает Planner;
 * - вызывает Validator.
 *
 * =========================================================
 */







/*
 * =========================================================
 * STATUS
 * =========================================================
 */


export const EXECUTION_RESULT_STATUS = {


    COMPLETED:
        "COMPLETED",


    FAILED:
        "FAILED",


    NEEDS_CLARIFICATION:
        "NEEDS_CLARIFICATION",


    NO_VERIFIED_RESULT:
        "NO_VERIFIED_RESULT"


};









/*
 * =========================================================
 * SAFE STRING
 * =========================================================
 */


function safeString(
    value
) {

    return String(
        value || ""
    )
        .trim();

}









/*
 * =========================================================
 * USED TOOLS
 * =========================================================
 */


function collectUsedTools(
    runResult
) {


    const results =

        Array.isArray(
            runResult?.results
        )

            ? runResult.results

            : [];



    return [

        ...new Set(

            results

                .map(
                    item =>
                        item?.tool
                )

                .filter(Boolean)

        )

    ];

}









/*
 * =========================================================
 * EXPERIENCE META
 * =========================================================
 */


function buildExperienceMeta(
    context
) {


    const skills =

        Array.isArray(
            context?.experience?.skills
        )

            ? context.experience.skills

            : [];



    return {


        used:

            context?.experience?.used === true,



        skills,



        skillIds:

            skills

                .map(

                    skill =>

                        typeof skill === "string"

                            ? skill

                            :

                            skill?.id ||
                            skill?.name

                )

                .filter(Boolean)


    };


}









/*
 * =========================================================
 * COMMON META
 * =========================================================
 */


function buildExecutionMeta(
    context
) {


    return {


        attempt:

            Number(
                context?.attempt || 0
            ),



        usedTools:

            collectUsedTools(
                context?.runResult
            ),



        experience:

            buildExperienceMeta(
                context
            ),



        planId:

            context?.plan?.id ||
            null


    };


}









/*
 * =========================================================
 * BASE
 * =========================================================
 */


function buildBaseResult(
    context
) {


    return {


        task:

            context?.task ||
            "",



        plan:

            context?.plan ||
            null,



        executionMeta:

            buildExecutionMeta(
                context
            )

    };

}









/*
 * =========================================================
 * COMPLETED
 * =========================================================
 */


export function buildCompletedResult(

    context,

    answerResult,

    validated = true

) {


    return {


        ...buildBaseResult(
            context
        ),



        success:

            true,



        status:

            EXECUTION_RESULT_STATUS
                .COMPLETED,



        verified:

            validated === true,



        answer:

        {

            text:

                answerResult?.text ||
                "",


            source:

                answerResult?.source ||
                "unknown"


        },



        failure:

            null


    };


}









/*
 * =========================================================
 * FAILED
 * =========================================================
 */


export function buildFailureResult(

    context,

    {

        stage = "execution",

        reason = "",

        failureType = "execution_failure"

    } = {}

) {


    return {


        ...buildBaseResult(
            context
        ),



        success:

            false,



        status:

            EXECUTION_RESULT_STATUS
                .FAILED,



        verified:

            false,



        answer:

            null,



        failure:

        {


            stage,


            type:
                failureType,


            reason:

                safeString(
                    reason
                )


        }


    };


}









/*
 * =========================================================
 * NEEDS CLARIFICATION
 * =========================================================
 */


export function buildClarificationResult(

    context,

    {

        stage = "execution",

        reason =
            "Требуется уточнение"

    } = {}

) {


    return {


        ...buildBaseResult(
            context
        ),



        success:

            false,



        status:

            EXECUTION_RESULT_STATUS
                .NEEDS_CLARIFICATION,



        verified:

            false,



        answer:

            null,



        clarification:

        {


            stage,


            reason:

                safeString(
                    reason
                )


        }



    };


}









/*
 * =========================================================
 * NO VERIFIED RESULT
 * =========================================================
 */


export function buildNoVerifiedResult(

    context,

    reason = ""

) {


    return {


        ...buildBaseResult(
            context
        ),



        success:

            false,



        status:

            EXECUTION_RESULT_STATUS
                .NO_VERIFIED_RESULT,



        verified:

            false,



        answer:

            null,



        failure:

        {


            stage:
                "verification",



            type:
                "no_verified_result",



            reason:

                safeString(
                    reason
                )


        }


    };


}









/*
 * =========================================================
 * RESULT HELPERS
 * =========================================================
 */


export function isCompletedResult(
    result
) {


    return (

        result?.status ===
        EXECUTION_RESULT_STATUS.COMPLETED

        &&

        result?.success === true

    );

}





export function isFailedResult(
    result
) {


    return (

        result?.status ===
        EXECUTION_RESULT_STATUS.FAILED

    );

}





export function isClarificationResult(
    result
) {


    return (

        result?.status ===
        EXECUTION_RESULT_STATUS
            .NEEDS_CLARIFICATION

    );

}
