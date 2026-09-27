/*
 * =========================================================
 * JESSICA RESULT META
 * =========================================================
 *
 * Сбор технической информации
 * для Execution Result.
 *
 *
 * НЕ:
 *
 * - создаёт Result;
 * - меняет Context;
 * - принимает решения.
 *
 * =========================================================
 */


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


export function collectUsedTools(

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


export function buildExperienceMeta(

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

            skills

                .map(

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


export function buildExecutionMeta(

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


export function buildHistoryMeta(

    context

) {


    return {


        failures:

            safeArray(

                context?.errors

            ),



        replans:

            safeArray(

                context?.replanHistory

            ),



        executionHistory:

            safeArray(

                context?.stepsHistory

            )


    };


}









/*
 * =========================================================
 * BASE RESULT
 * =========================================================
 */


export function buildBaseResult(

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
