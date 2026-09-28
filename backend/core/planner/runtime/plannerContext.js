/*
 * =========================================================
 * JESSICA PLANNER CONTEXT
 * =========================================================
 *
 * Подготавливает PlanningContext
 * для запуска Planner.
 *
 * =========================================================
 */


import {
    normalizePlanningContext
} from "../planningContext.js";


/*
 * =========================================================
 * BASE CONTEXT
 * =========================================================
 */


export function buildPlannerContext(
    context
) {


    const normalized =
        normalizePlanningContext(
            context
        );


    return {

        ...normalized,

        metadata: {

            ...(normalized.metadata || {}),

            planner: {

                version:
                    "4",

                createdAt:
                    new Date()
                        .toISOString()

            }

        }

    };

}


/*
 * =========================================================
 * ATTEMPT CONTEXT
 * =========================================================
 */


export function buildPlannerAttemptContext(
    baseContext,
    attempt
) {


    return {

        ...baseContext,

        metadata: {

            ...(baseContext?.metadata || {}),

            plannerAttempt:
                attempt

        }

    };

}
