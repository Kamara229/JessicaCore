/*
 * =========================================================
 * JESSICA PLANNER PLAN RESULT
 * =========================================================
 *
 * Добавляет служебные metadata
 * к готовому Execution Plan.
 *
 * =========================================================
 */


export function enrichPlan(

    plan,

    context,

    attempt

) {


    return {

        ...plan,

        metadata: {

            ...(plan?.metadata || {}),

            plannerAttempt:
                attempt,

            generatedAt:
                new Date()
                    .toISOString()

        },

        experience: {

            ...(plan?.experience || {}),

            available:
                Boolean(
                    context?.experience
                )

        }

    };

}
