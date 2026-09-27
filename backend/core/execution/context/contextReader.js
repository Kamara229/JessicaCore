/*
 * =========================================================
 * JESSICA CONTEXT READER v2
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
 * - results;
 * - plans.
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


    const number =

        Number(value);



    return Number.isFinite(number)

        ?

        number

        :

        0;

}









/*
 * =========================================================
 * ID
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
 * RESULTS
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









export function getTerminalResult(

    context

) {


    return context?.terminalResult || null;

}









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
 * STEPS
 * =========================================================
 */


export function getExecutionHistory(

    context

) {


    if(

        Array.isArray(context?.stepsHistory)

        &&

        context.stepsHistory.length > 0

    ){

        return context.stepsHistory;

    }



    return safeArray(

        context?.executionHistory

    );

}









/*
 * =========================================================
 * PLANS
 * =========================================================
 */


export function getInitialPlan(

    context

) {


    return context?.initialPlan || null;

}









export function getCurrentPlan(

    context

) {


    return context?.plan || null;

}









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
