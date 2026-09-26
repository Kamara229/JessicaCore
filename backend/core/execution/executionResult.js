/*
 * =========================================================
 * JESSICA EXECUTION RESULT v5
 * =========================================================
 *
 * Единый контракт результата Execution.
 *
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









function safeString(
    value
) {

    return String(
        value || ""
    )
    .trim();

}









function safeArray(
    value
) {

    return Array.isArray(value)

        ? value

        : [];

}









/*
 * =========================================================
 * TOOLS
 * =========================================================
 */


function collectUsedTools(
    context
) {


    const results =

        safeArray(
            context?.runResult?.results
        );



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
 * EXPERIENCE
 * =========================================================
 */


function buildExperienceMeta(
    context
) {


    const skills =

        safeArray(
            context?.experience?.skills
        );



    return {


        used:

            context?.experience?.found === true
            ||
            context?.experience?.used === true,



        source:

            context?.experience?.source ||
            null,



        confidence:

            Number(
                context?.experience?.confidence || 0
            ),



        skills,



        skillIds:

            skills.map(

                skill =>

                    typeof skill === "string"

                    ?

                    skill

                    :

                    skill?.id ||
                    skill?.name

            )

            .filter(Boolean)


    };

}









/*
 * =========================================================
 * EXECUTION META
 * =========================================================
 */


function buildExecutionMeta(
    context
) {


    return {


        executionId:

            context?.executionId ||
            null,



        traceId:

            context?.trace?.id ||
            null,



        attempts:

            Number(
                context?.attempt || 0
            ),



        retryCount:

            Number(
                context?.retryCount || 0
            ),



        replanCount:

            Number(
                context?.replanCount || 0
            ),



        usedTools:

            collectUsedTools(
                context
            ),



        experience:

            buildExperienceMeta(
                context
            )

    };

}









/*
 * =========================================================
 * HISTORY
 * =========================================================
 */


function buildHistoryMeta(
    context
) {


    return {


        failures:

            safeArray(
                context?.errors
            ),



        replans:

            safeArray(
                context?.replans
            ),



        executionHistory:

            safeArray(
                context?.executionHistory
            )


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

            context?.task || "",



        initialPlan:

            context?.initialPlan ||
            null,



        currentPlan:

            context?.plan ||
            null,



        executionMeta:

            buildExecutionMeta(
                context
            ),



        history:

            buildHistoryMeta(
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

    verified = true

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

            verified === true,



        answer:

        {

            text:

                safeString(
                    answerResult?.text
                ),



            source:

                answerResult?.source ||
                "unknown"


        },



        failure:

            null,


        clarification:

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

        stage="execution",

        reason="Не удалось выполнить задачу",

        failureType="execution_failure"

    }={}

) {


    return {


        ...buildBaseResult(
            context
        ),



        success:false,



        status:

            EXECUTION_RESULT_STATUS
                .FAILED,



        verified:false,



        answer:null,



        clarification:null,



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
 * CLARIFICATION
 * =========================================================
 */


export function buildClarificationResult(

    context,

    {

        stage="execution",

        reason="Требуется уточнение"

    }={}

) {


    return {


        ...buildBaseResult(
            context
        ),



        success:false,



        status:

            EXECUTION_RESULT_STATUS
                .NEEDS_CLARIFICATION,



        verified:false,



        answer:null,



        failure:null,



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
 * NO VERIFIED
 * =========================================================
 */


export function buildNoVerifiedResult(

    context,

    {

        stage="verification",

        reason="Не удалось подтвердить результат",

        failureType="no_verified_result"

    }={}

) {


    return {


        ...buildBaseResult(
            context
        ),



        success:true,



        status:

            EXECUTION_RESULT_STATUS
                .NO_VERIFIED_RESULT,



        verified:false,



        answer:null,



        clarification:null,



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
 * HELPERS
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



export function isNoVerifiedResult(
    result
) {

    return (

        result?.status ===
        EXECUTION_RESULT_STATUS
        .NO_VERIFIED_RESULT

    );

}
