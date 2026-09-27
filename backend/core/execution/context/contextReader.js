/*
 * =========================================================
 * JESSICA CONTEXT READER v1
 * =========================================================
 *
 * Безопасное чтение Execution Context.
 *
 *
 * Отвечает:
 *
 * - counters;
 * - execution history;
 * - failures;
 * - experience;
 * - run results.
 *
 *
 * НЕ:
 *
 * - изменяет Context;
 * - создаёт Context;
 * - управляет Retry/Replan.
 *
 * =========================================================
 */







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
 * SAFE NUMBER
 * =========================================================
 */


function safeNumber(

    value

) {


    return Number(

        value || 0

    );

}









/*
 * =========================================================
 * EXECUTION ID
 * =========================================================
 */


export function getExecutionId(

    context

) {


    return (

        context?.executionId

        ||

        null

    );

}









/*
 * =========================================================
 * TRACE ID
 * =========================================================
 */


export function getTraceId(

    context

) {


    return (

        context?.trace?.id

        ||

        null

    );

}









/*
 * =========================================================
 * COUNTERS
 * =========================================================
 */


export function getExecutionCounters(

    context

) {


    return {


        attempt:

            safeNumber(

                context?.attempt

            ),



        retryCount:

            safeNumber(

                context?.retryCount

            ),



        replanCount:

            safeNumber(

                context?.replanCount

            )


    };

}









/*
 * =========================================================
 * EXPERIENCE
 * =========================================================
 */


export function getExperience(

    context

) {


    return {


        used:

            context?.experience?.used === true,



        found:

            context?.experience?.found === true,



        source:

            context?.experience?.source

            ||

            null,



        confidence:

            safeNumber(

                context?.experience?.confidence

            ),



        skills:

            safeArray(

                context?.experience?.skills

            ),



        context:

            context?.experience?.context

            ||

            null


    };

}









/*
 * =========================================================
 * RUN RESULTS
 * =========================================================
 */


export function getRunResults(

    context

) {


    return safeArray(

        context?.runResult?.results

        ||

        context?.runResult?.result?.results

    );

}









/*
 * =========================================================
 * FAILURES
 * =========================================================
 */


export function getExecutionFailures(

    context

) {


    return safeArray(

        context?.errors

    );

}









/*
 * =========================================================
 * REPLANS
 * =========================================================
 */


export function getExecutionReplans(

    context

) {


    return safeArray(

        context?.replanHistory

    );

}









/*
 * =========================================================
 * STEPS HISTORY
 * =========================================================
 */


export function getExecutionHistory(

    context

) {


    return safeArray(

        context?.stepsHistory

        ||

        context?.executionHistory

    );

}









/*
 * =========================================================
 * PLANS
 * =========================================================
 */


export function getPlans(

    context

) {


    return {


        initialPlan:

            context?.initialPlan

            ||

            null,



        currentPlan:

            context?.plan

            ||

            null


    };

}
