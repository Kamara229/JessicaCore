import {
    createPlan
} from "./planner.js";


/*
 * =========================================================
 * JESSICA REPLANNER
 * =========================================================
 *
 * Создаёт новый маршрут после неудачного выполнения.
 *
 *
 * Поток:
 *
 * Execution Cycle
 *       ↓
 * Failure Context
 *       ↓
 * Replanner
 *       ↓
 * createPlan()
 *       ↓
 * Planner
 *
 *
 * Replanner НЕ:
 *
 * - вызывает AI напрямую;
 * - строит prompt;
 * - знает инструменты;
 * - валидирует план.
 *
 *
 * Использует тот же Planner,
 * что и первоначальное планирование.
 *
 * =========================================================
 */



/*
 * =========================================================
 * BUILD RETRY METADATA
 * =========================================================
 */


function buildRetryMetadata(
    failure
) {

    return {

        stage:
            failure?.stage || "",


        failureType:
            failure?.failureType || "",


        reason:
            failure?.reason || "",


        timestamp:
            new Date()
                .toISOString()

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


    const retryMetadata =
        buildRetryMetadata(
            failure
        );



    return {


        ...(context || {}),



        metadata: {


            ...(context?.metadata || {}),


            retry:
                retryMetadata

        },



        instructions: [

            ...(context?.instructions || []),


            "Предыдущий маршрут выполнения не дал корректный результат.",


            "Используй информацию об ошибке при построении нового маршрута.",


            "Измени стратегию, если предыдущий подход оказался неэффективным."

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
     * Сохраняем минимальную историю.
     *
     * Не кладём полный результат выполнения,
     * чтобы не раздувать prompt.
     */


    retryContext.metadata = {


        ...(retryContext.metadata || {}),



        previousAttempt: {


            intent:
                previousPlan?.intent || "",


            stepsCount:
                Array.isArray(
                    previousPlan?.steps
                )
                    ? previousPlan.steps.length
                    : 0,


            failedTools:
                Array.isArray(
                    previousRunResult?.results
                )
                    ? previousRunResult.results
                        .filter(
                            item =>
                                item?.success === false
                        )
                        .map(
                            item =>
                                item.tool
                        )
                    : []

        }

    };



    console.log(

        "Jessica Replanner context:",

        JSON.stringify(

            retryContext.metadata

        )

    );



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
                    "Не удалось создать новый план"


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
