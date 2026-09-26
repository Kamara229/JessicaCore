/*
 * =========================================================
 * JESSICA REPLAN COORDINATOR v5
 * =========================================================
 *
 * Координатор перестроения Execution Plan.
 *
 *
 * Flow:
 *
 * Replan Decision
 *        ↓
 * Build Replan Request
 *        ↓
 * Replanner
 *        ↓
 * Alternative Plan
 *        ↓
 * Apply Context
 *
 *
 * НЕ:
 *
 * - принимает решение о Replan;
 * - анализирует ошибки;
 * - выполняет Tools;
 * - запускает Execution;
 * - изменяет Experience.
 *
 * =========================================================
 */


import {
    replanTask
} from "../replanner.js";









/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


const MAX_REPLAN_COUNT = 3;









/*
 * =========================================================
 * STRING
 * =========================================================
 */


function safeString(
    value
) {


    return String(
        value || ""
    )
    .trim();

}









/*
 * =========================================================
 * NORMALIZE FAILURE
 * =========================================================
 */


function normalizeFailure(
    failure
) {


    if (
        !failure ||
        typeof failure !== "object"
    ) {


        return {


            stage:
                "execution",



            failureType:
                "unknown",



            category:
                "execution",



            reason:
                "Причина перестроения неизвестна"


        };

    }





    return {


        stage:

            safeString(
                failure.stage
            )
            ||
            "execution",



        failureType:

            safeString(
                failure.failureType
            )
            ||
            "execution-error",



        category:

            safeString(
                failure.category
            )
            ||
            "execution",



        reason:

            safeString(
                failure.reason
            )
            ||
            "Текущий план не дал результат"


    };

}









/*
 * =========================================================
 * BUILD REQUEST
 * =========================================================
 */


function buildReplanRequest(

    context,

    failure

) {


    return {


        task:

            context?.task ||
            "",



        previousPlan:

            context?.plan ||
            null,



        failure,



        attempt:

            Number(
                context?.attempt || 0
            ),



        replanCount:

            Number(
                context?.replanCount || 0
            ),



        runResult:

            context?.runResult ||
            null,



        planningContext:

            context?.planningContext ||
            {}

    };

}









/*
 * =========================================================
 * CREATE ALTERNATIVE PLAN
 * =========================================================
 */


export async function createAlternativePlan(

    context,

    failure

) {



    const normalizedFailure =

        normalizeFailure(
            failure
        );






    const request =

        buildReplanRequest(

            context,

            normalizedFailure

        );








    if (

        request.replanCount >= MAX_REPLAN_COUNT

    ) {


        return {


            success:false,


            reason:
                "Достигнут лимит перестроений"


        };

    }








    console.log(

        "Jessica Replan Coordinator:",

        {

            attempt:
                request.attempt,


            replanCount:
                request.replanCount,


            failureType:
                normalizedFailure.failureType,


            reason:
                normalizedFailure.reason


        }

    );








    try {


        const result =

            await replanTask(

                request.task,


                request.previousPlan,


                request.failure,


                request.runResult,


                {


                    ...request.planningContext,



                    replanContext:

                    {

                        failure:

                            normalizedFailure,



                        previousPlan:

                            request.previousPlan,



                        attempt:

                            request.attempt

                    }

                }

            );








        if (

            !result ||

            result.success !== true ||

            !result.plan

        ) {


            return {


                success:false,


                reason:

                    result?.reason ||

                    "Replanner не создал альтернативный план"


            };

        }








        return {


            success:true,


            plan:

                result.plan,



            planningContext:

                result.context &&

                typeof result.context === "object"

                    ?

                    result.context

                    :

                    context?.planningContext || {}



        };





    } catch(error) {



        console.error(

            "Jessica Replan Coordinator error:",

            error

        );



        return {


            success:false,


            reason:

                error?.message ||

                "Ошибка перестроения плана"


        };


    }


}









/*
 * =========================================================
 * APPLY PLAN
 * =========================================================
 */


export function applyAlternativePlan(

    context,

    alternative

) {


    if (

        !context ||

        !alternative?.success ||

        !alternative.plan

    ) {


        return false;

    }








    /*
     * Новый execution route
     */


    context.plan =

        alternative.plan;






    /*
     * Новый planning context
     */


    if (

        alternative.planningContext

    ) {


        context.planningContext =

            alternative.planningContext;


    }








    /*
     * Старый результат больше
     * не относится к новому маршруту
     */


    context.runResult =
        null;


    context.answerResult =
        null;


    context.validationResult =
        null;








    context.replanCount =

        Number(
            context.replanCount || 0
        )

        +

        1;








    if (
        !Array.isArray(
            context.replans
        )
    ) {

        context.replans = [];

    }






    context.replans.push({

        planCreated:

            new Date()
            .toISOString(),



        reason:

            context
                ?.failure
                ?.reason ||
            "execution failure"


    });








    return true;

}









/*
 * =========================================================
 * VALIDATION
 * =========================================================
 */


export function canReplan(
    context
) {


    return (

        Number(
            context?.replanCount || 0
        )

        <

        MAX_REPLAN_COUNT

    );

}
