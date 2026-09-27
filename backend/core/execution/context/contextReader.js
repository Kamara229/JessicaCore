/*
 * =========================================================
 * JESSICA CONTEXT READER v3
 * =========================================================
 *
 * Безопасное чтение Execution Context.
 *
 *
 * Отвечает:
 *
 * - identity;
 * - state;
 * - counters;
 * - execution history;
 * - failures;
 * - attempts;
 * - replans;
 * - experience;
 * - results;
 * - plans.
 *
 *
 * НЕ:
 *
 * - изменяет Context;
 * - создаёт Context;
 * - управляет Retry;
 * - управляет Replan;
 * - выполняет Execution.
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

        [
            ...value
        ]

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


    const number =

        Number(

            value

        );



    return Number.isFinite(number)

        ?

        number

        :

        0;

}









/*
 * =========================================================
 * IDENTITY
 * =========================================================
 */


export function getExecutionId(

    context

) {


    return context?.executionId || null;

}









export function getTraceId(

    context

) {


    return context?.trace?.id || null;

}









/*
 * =========================================================
 * STATE
 * =========================================================
 */


export function getExecutionState(

    context

) {


    return {


        state:

            context?.state || null,



        status:

            context?.status || null


    };

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

            context?.experience?.source || null,



        confidence:

            safeNumber(

                context?.experience?.confidence

            ),



        skills:

            safeArray(

                context?.experience?.skills

            ),



        context:

            context?.experience?.context || null


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
 * TERMINAL RESULT
 * =========================================================
 */


export function getTerminalResult(

    context

) {


    return context?.terminalResult || null;

}









/*
 * =========================================================
 * RESULT HISTORY
 * =========================================================
 */


export function getResultHistory(

    context

) {


    return safeArray(

        context?.resultHistory

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
 * ATTEMPTS
 * =========================================================
 */


export function getExecutionAttempts(

    context

) {


    return safeArray(

        context?.attempts

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
 * EXECUTION HISTORY
 * =========================================================
 */


export function getExecutionHistory(

    context

) {


    return safeArray(

        context?.stepsHistory

    );

}









/*
 * =========================================================
 * INITIAL PLAN
 * =========================================================
 */


export function getInitialPlan(

    context

) {


    return context?.initialPlan || null;

}









/*
 * =========================================================
 * CURRENT PLAN
 * =========================================================
 */


export function getCurrentPlan(

    context

) {


    return context?.plan || null;

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

            getInitialPlan(

                context

            ),



        currentPlan:

            getCurrentPlan(

                context

            )


    };

}
