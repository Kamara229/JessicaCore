import {
    buildRetryHistory
} from "./retryHistory.js";

import {
    buildRetryInstructions
} from "./retryInstructions.js";


/*
 * =========================================================
 * JESSICA REPLANNER
 * RETRY CONTEXT
 * =========================================================
 *
 * Собирает PlanningContext для повторного планирования.
 *
 *
 * Flow:
 *
 * Previous PlanningContext
 *        +
 * Previous Plan
 *        +
 * Previous Run Result
 *        +
 * Failure
 *        ↓
 * Retry History
 *        ↓
 * Retry Instructions
 *        ↓
 * Updated PlanningContext
 *
 *
 * Этот модуль НЕ:
 *
 * - вызывает Planner;
 * - вызывает AI;
 * - выполняет инструменты;
 * - принимает решение о retry;
 * - хранит полный execution result.
 *
 * =========================================================
 */


/*
 * =========================================================
 * SAFE STRING
 * =========================================================
 */


function safeString(
    value
) {

    return typeof value === "string"
        ? value.trim()
        : "";

}


/*
 * =========================================================
 * NORMALIZE CONTEXT
 * =========================================================
 */


function normalizeBaseContext(
    planningContext
) {

    if (
        !planningContext ||
        typeof planningContext !== "object" ||
        Array.isArray(planningContext)
    ) {

        return {};

    }


    return {

        ...planningContext,

        metadata: {

            ...(
                planningContext.metadata &&
                typeof planningContext.metadata === "object" &&
                !Array.isArray(
                    planningContext.metadata
                )

                    ? planningContext.metadata

                    : {}
            )

        },

        instructions:

            Array.isArray(
                planningContext.instructions
            )

                ? [
                    ...planningContext.instructions
                ]

                : []

    };

}


/*
 * =========================================================
 * BUILD RETRY METADATA
 * =========================================================
 */


function buildRetryMetadata(
    failureResult
) {

    return {

        stage:
            safeString(
                failureResult?.stage
            ),

        failureType:
            safeString(
                failureResult?.failureType
            ),

        reason:
            safeString(
                failureResult?.reason ||
                failureResult?.text
            ),

        timestamp:
            new Date()
                .toISOString()

    };

}


/*
 * =========================================================
 * REPLAN COUNT
 * =========================================================
 */


function getNextReplanCount(
    planningContext
) {

    const current =
        Number(
            planningContext
                ?.metadata
                ?.replanCount
        );


    if (
        !Number.isFinite(current) ||
        current < 0
    ) {

        return 1;

    }


    return (
        Math.floor(current) + 1
    );

}


/*
 * =========================================================
 * BUILD RETRY CONTEXT
 * =========================================================
 */


export function buildRetryContext({

    planningContext = {},

    previousPlan = null,

    previousRunResult = null,

    failureResult = {}

} = {}) {


    /*
     * =====================================================
     * BASE CONTEXT
     * =====================================================
     */


    const baseContext =
        normalizeBaseContext(
            planningContext
        );


    /*
     * =====================================================
     * RETRY HISTORY
     * =====================================================
     */


    const retryHistory =
        buildRetryHistory({

            planningContext:
                baseContext,

            previousPlan,

            previousRunResult

        });


    /*
     * =====================================================
     * RETRY INSTRUCTIONS
     * =====================================================
     */


    const instructions =
        buildRetryInstructions({

            planningContext:
                baseContext,

            failureResult,

            retryHistory

        });


    /*
     * =====================================================
     * RETRY METADATA
     * =====================================================
     */


    const retry =
        buildRetryMetadata(
            failureResult
        );


    const replanCount =
        getNextReplanCount(
            baseContext
        );


    /*
     * =====================================================
     * RESULT
     * =====================================================
     *
     * Сохраняем существующие:
     *
     * - Experience;
     * - sourceRules;
     * - constraints;
     * - plannerHints;
     * - другие metadata.
     *
     *
     * Обновляем только данные,
     * относящиеся к текущему Replan.
     *
     * =====================================================
     */


    return {

        ...baseContext,


        instructions,


        metadata: {

            ...(baseContext.metadata || {}),


            /*
             * Текущая причина Replan.
             */

            retry,


            /*
             * Компактное описание
             * непосредственно предыдущей попытки.
             */

            previousAttempt:
                retryHistory.previousAttempt,


            /*
             * Накопительная история
             * поисковых запросов.
             */

            triedSearchQueries:
                retryHistory.triedSearchQueries,


            /*
             * Сколько раз Execution Cycle
             * уже запрашивал новый маршрут.
             */

            replanCount

        }

    };

}
