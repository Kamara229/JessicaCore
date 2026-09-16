import {
    createPlan
} from "./planner.js";


/*
 * =========================================================
 * JESSICA REPLANNER
 * =========================================================
 *
 * Отвечает только за создание нового маршрута
 * после неудачного выполнения.
 *
 *
 * НЕ:
 *
 * - вызывает AI напрямую;
 * - создает prompt;
 * - знает инструменты;
 * - валидирует JSON.
 *
 *
 * Использует основной Planner:
 *
 * Replanner
 *      ↓
 * createPlan()
 *      ↓
 * Planner
 *
 *
 * Сохраняет:
 *
 * - Experience;
 * - PlanningContext;
 * - ограничения;
 * - историю ошибки.
 *
 * =========================================================
 */


/*
 * =========================================================
 * BUILD RETRY CONTEXT
 * =========================================================
 */


function buildRetryContext(
    failure
) {

    return {

        retry: {

            stage:
                failure?.stage || "",


            failureType:
                failure?.failureType || "",


            reason:
                failure?.reason || "",


            shouldRetry:
                true

        }

    };

}



/*
 * =========================================================
 * MERGE CONTEXT
 * =========================================================
 */


function mergePlanningContext(
    context,
    failure
) {


    return {

        ...(context || {}),


        metadata: {

            ...(context?.metadata || {}),


            retry:

                {

                    stage:
                        failure?.stage || "",


                    failureType:
                        failure?.failureType || "",


                    reason:
                        failure?.reason || ""

                }

        },


        instructions: [

            ...(context?.instructions || []),

            "Предыдущий маршрут выполнения завершился ошибкой.",

            "Используй причину ошибки при построении нового плана.",

            "Не повторяй неэффективный маршрут без изменений."

        ]

    };

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


    const retryContext =
        mergePlanningContext(

            planningContext,

            failureResult

        );



    /*
     * Добавляем историю ошибки
     */


    retryContext.metadata = {

        ...(retryContext.metadata || {}),


        previousPlan,

        previousRunResult

    };



    /*
     * =====================================================
     * CREATE NEW PLAN
     * =====================================================
     */


    try {


        const result =
            await createPlan(

                task,

                retryContext

            );



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



        console.log(

            "Jessica Replanner new plan:",

            JSON.stringify(
                result.plan
            )

        );



        return {

            success:
                true,


            plan:
                result.plan,


            context:
                retryContext

        };


    } catch(error) {


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
