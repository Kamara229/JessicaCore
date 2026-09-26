/*
 * =========================================================
 * JESSICA EXECUTION RESULT
 * =========================================================
 *
 * Формирует стандартизированные результаты
 * Execution Cycle.
 *
 *
 * Поддерживает:
 *
 * - COMPLETED;
 * - FAILED;
 * - NEEDS_CLARIFICATION;
 * - NO_VERIFIED_RESULT.
 *
 *
 * Добавляет:
 *
 * - Experience metadata;
 * - Execution metadata;
 * - Learning context.
 *
 *
 * НЕ:
 *
 * - выполняет инструменты;
 * - вызывает Planner;
 * - вызывает Validator.
 *
 * =========================================================
 */






/*
 * =========================================================
 * USED TOOLS
 * =========================================================
 */


function collectUsedTools(
    taskRunResult
) {


    const results =

        Array.isArray(
            taskRunResult?.results
        )

            ? taskRunResult.results

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


function buildExperienceData(
    context
) {


    return {


        used:

            context?.experience?.used === true,



        skills:

            Array.isArray(
                context?.experience?.skills
            )

                ? context.experience.skills

                : [],



        skillIds:

            Array.isArray(
                context?.experience?.skills
            )

                ? context.experience.skills

                    .map(

                        skill =>

                            skill?.id ||
                            skill

                    )

                    .filter(Boolean)

                : []

    };

}









/*
 * =========================================================
 * COMMON DATA
 * =========================================================
 */


function buildCommonData(
    context
) {


    return {


        plan:

            context?.plan || null,




        planningContext:

            context?.planningContext || {},





        experience:

            buildExperienceData(
                context
            ),





        toolResults:

            context?.runResult?.results || [],





        usedTools:

            collectUsedTools(
                context?.runResult
            ),





        attempt:

            Number(
                context?.attempt || 0
            ),






        executionMeta:

        {

            attempt:

                Number(
                    context?.attempt || 0
                ),



            experienceUsed:

                context?.experience?.used === true

        }



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

    validated

) {


    const isValidated =

        validated === true;




    return {


        success:

            true,



        status:

            "COMPLETED",



        resultType:

            "result",



        verified:

            isValidated,



        validated:

            isValidated,



        validationStatus:

            isValidated

                ? "passed"

                : "skipped",





        result:

            answerResult?.text || "",





        answerSource:

            answerResult?.source ||

            "unknown",






        ...buildCommonData(
            context
        )

    };

}









/*
 * =========================================================
 * FAILURE
 * =========================================================
 */


export function buildFailureResult(

    context,

    {

        stage = "execution",

        reason = "Не удалось выполнить задачу",

        failureType = null

    } = {}

) {


    return {


        success:

            false,



        status:

            "FAILED",



        resultType:

            "failure",



        verified:

            false,



        shouldRetry:

            false,



        stage,



        failureType,



        result:

            reason,



        ...buildCommonData(
            context
        )

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
            "Для выполнения задачи требуется уточнение"

    } = {}

) {


    return {


        success:

            false,



        status:

            "NEEDS_CLARIFICATION",



        resultType:

            "needs_clarification",



        verified:

            false,



        shouldRetry:

            false,



        needsClarification:

            true,



        stage,



        result:

            reason,



        ...buildCommonData(
            context
        )

    };

}









/*
 * =========================================================
 * NO VERIFIED RESULT
 * =========================================================
 */


export function buildNoVerifiedResult(

    context,

    {

        message =
            "Не удалось подтвердить достоверный результат по доступным источникам.",


        reason =
            "",


        stage =
            "search",


        failureType =
            null

    } = {}

) {


    return {


        success:

            true,



        status:

            "COMPLETED",



        resultType:

            "no_verified_result",



        verified:

            false,



        validated:

            false,



        validationStatus:

            "not_applicable",



        shouldRetry:

            false,



        stage,



        failureType,



        result:

            message,



        reason,



        answerSource:

            "execution",




        ...buildCommonData(
            context
        )

    };

}
