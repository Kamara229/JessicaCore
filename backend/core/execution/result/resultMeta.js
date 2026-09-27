/*
 * =========================================================
 * JESSICA RESULT META v3
 * =========================================================
 *
 * Формирование metadata для Execution Result.
 *
 *
 * Использует:
 *
 * Context Reader API
 *
 *
 * НЕ:
 *
 * - читает Context напрямую;
 * - меняет Context;
 * - принимает решения.
 *
 * =========================================================
 */


import {
    getExecutionId,
    getTraceId,
    getExecutionCounters,
    getExperience,
    getRunResults,
    getExecutionFailures,
    getExecutionReplans,
    getExecutionHistory,
    getPlans
} from "../context/contextReader.js";









/*
 * =========================================================
 * SAFE ARRAY
 * =========================================================
 */


function safeArray(

    value

) {


    return Array.isArray(value)

        ?

        value

        :

        [];

}









/*
 * =========================================================
 * USED TOOLS
 * =========================================================
 */


export function collectUsedTools(

    context

) {


    const results =

        safeArray(

            getRunResults(

                context

            )

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
 * EXPERIENCE META
 * =========================================================
 */


export function buildExperienceMeta(

    context

) {


    const experience =

        getExperience(

            context

        );



    const skills =

        safeArray(

            experience.skills

        );



    return {


        used:

            experience.used

            ||

            experience.found,



        source:

            experience.source,



        confidence:

            experience.confidence,



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


    const counters =

        getExecutionCounters(

            context

        );



    return {


        executionId:

            getExecutionId(

                context

            ),



        traceId:

            getTraceId(

                context

            ),



        attempt:

            counters.attempt,



        retryCount:

            counters.retryCount,



        replanCount:

            counters.replanCount,



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
 * HISTORY META
 * =========================================================
 */


export function buildHistoryMeta(

    context

) {


    return {


        failures:

            getExecutionFailures(

                context

            ),



        replans:

            getExecutionReplans(

                context

            ),



        executionHistory:

            getExecutionHistory(

                context

            )


    };

}









/*
 * =========================================================
 * BASE RESULT DATA
 * =========================================================
 */


export function buildBaseResult(

    context

) {


    const plans =

        getPlans(

            context

        );



    return {


        resultType:

            "EXECUTION_RESULT",



        task:

            context?.task || "",



        initialPlan:

            plans.initialPlan,



        currentPlan:

            plans.currentPlan,



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
