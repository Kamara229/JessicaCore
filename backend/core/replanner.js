import {
    createPlan
} from "./planner.js";

import {
    buildRetryContext
} from "./replanner/retryContext.js";


/*
 * =========================================================
 * JESSICA REPLANNER
 * =========================================================
 *
 * Центральный координатор повторного планирования.
 *
 *
 * Flow:
 *
 * Failure
 *   ↓
 * buildRetryContext()
 *   ↓
 * createPlan()
 *   ↓
 * New Plan
 *
 *
 * Replanner НЕ:
 *
 * - вызывает AI напрямую;
 * - строит prompt;
 * - анализирует историю самостоятельно;
 * - выполняет инструменты;
 * - валидирует план самостоятельно;
 * - хранит полный Execution Result.
 *
 *
 * Вспомогательная логика вынесена в:
 *
 * replanner/
 *
 * - retryHistory.js
 * - retryInstructions.js
 * - retryContext.js
 *
 * =========================================================
 */


/*
 * =========================================================
 * LOG RETRY CONTEXT
 * =========================================================
 */


function logRetryContext(
    retryContext
) {

    const metadata =
        retryContext?.metadata || {};


    console.log(

        "Jessica Replanner context:",

        JSON.stringify({

            replanCount:
                metadata.replanCount || 0,

            retry:
                metadata.retry || null,

            previousAttempt:
                metadata.previousAttempt || null,

            triedSearchQueries:
                metadata.triedSearchQueries || []

        })

    );

}


/*
 * =========================================================
 * REPLAN TASK
 * =========================================================
 */


export async function replanTask(

    task,

    previousPlan,

    failureResult,

    previousRunResult,

    planningContext = {}

) {


    /*
     * =====================================================
     * 1. BUILD RETRY CONTEXT
     * =====================================================
     */


    const retryContext =
        buildRetryContext({

            planningContext,

            previousPlan,

            previousRunResult,

            failureResult

        });


    logRetryContext(
        retryContext
    );


    /*
     * =====================================================
     * 2. CREATE NEW PLAN
     * =====================================================
     */


    try {


        const result =
            await createPlan(

                task,

                retryContext

            );


        /*
         * =================================================
         * PLANNER FAILURE
         * =================================================
         */


        if (
            !result?.success ||
            !result?.plan
        ) {

            return {

                success:
                    false,

                reason:
                    result?.text ||
                    "Planner не смог создать альтернативный маршрут"

            };

        }


        /*
         * =================================================
         * LOG NEW PLAN
         * =================================================
         */


        console.log(

            "Jessica Replanner new plan:",

            JSON.stringify(
                result.plan
            )

        );


        /*
         * =================================================
         * EFFECTIVE CONTEXT
         * =================================================
         *
         * Planner может нормализовать или дополнить
         * PlanningContext.
         *
         * Поэтому возвращаем result.context.
         *
         * Если Planner его не вернул —
         * используем retryContext.
         *
         * =================================================
         */


        const effectiveContext =
            result?.context &&
            typeof result.context === "object" &&
            !Array.isArray(
                result.context
            )

                ? result.context

                : retryContext;


        /*
         * =================================================
         * SUCCESS
         * =================================================
         */


        return {

            success:
                true,

            plan:
                result.plan,

            context:
                effectiveContext

        };


    } catch (error) {


        /*
         * =================================================
         * UNEXPECTED ERROR
         * =================================================
         */


        console.error(

            "Jessica Replanner error:",

            error

        );


        return {

            success:
                false,

            reason:
                error?.message ||
                "Ошибка Replanner"

        };

    }

}
