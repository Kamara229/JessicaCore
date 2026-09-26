/*
 * =========================================================
 * JESSICA REPLAN COORDINATOR v6
 * =========================================================
 *
 * Координатор перестроения Execution Plan.
 *
 *
 * Flow:
 *
 * Replan Decision
 *        ↓
 * Build Request
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
 * - принимает решение;
 * - анализирует ошибки;
 * - выполняет Tools;
 * - запускает Execution.
 *
 * =========================================================
 */


import {
    replanTask
} from "../replanner.js";









const MAX_REPLAN_COUNT = 3;









function safeString(
    value
) {

    return String(
        value || ""
    )
    .trim();

}









function normalizeFailure(
    failure
) {


    return {


        stage:

            safeString(
                failure?.stage
            )
            ||
            "execution",



        failureType:

            safeString(
                failure?.failureType
            )
            ||
            "execution-error",



        category:

            safeString(
                failure?.category
            )
            ||
            "execution",



        reason:

            safeString(
                failure?.reason
            )
            ||
            "План выполнения оказался неэффективным"


    };

}









function buildReplanRequest(

    context,

    failure

) {


    return {


        task:

            context.task,



        previousPlan:

            context.plan,



        failure,



        attempt:

            context.attempt || 0,



        retryCount:

            context.retryCount || 0,



        replanCount:

            context.replanCount || 0,



        runResult:

            context.runResult,



        planningContext:

            context.planningContext || {}

    };

}









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
                "Лимит перестроений исчерпан"


        };

    }








    console.log(

        "Jessica Replan:",

        {

            attempt:
                request.attempt,


            replanCount:
                request.replanCount,


            failure:
                normalizedFailure

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

                        previousPlan:

                            request.previousPlan,



                        failure:

                            normalizedFailure,



                        attempt:

                            request.attempt,



                        retryCount:

                            request.retryCount

                    }


                }

            );








        if (

            !result?.success ||

            !result?.plan

        ) {


            return {


                success:false,


                reason:

                    result?.reason ||

                    "Новый план не создан"


            };

        }








        return {


            success:true,


            plan:

                result.plan,



            failure:

                normalizedFailure,



            planningContext:

                result.context &&

                typeof result.context === "object"

                    ?

                    result.context

                    :

                    request.planningContext



        };





    } catch(error) {


        return {


            success:false,


            reason:

                error?.message ||

                "Ошибка Replanner"


        };


    }


}









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








    const previousPlan =
        context.plan;








    context.plan =

        alternative.plan;








    if (
        alternative.planningContext
    ) {


        context.planningContext =

            alternative.planningContext;


    }








    /*
     * Старый маршрут больше невалиден
     */


    context.runResult =
        null;


    context.answerResult =
        null;


    context.validationResult =
        null;








    /*
     * Новый маршрут начинает
     * новый retry цикл
     */


    context.retryCount =
        0;








    /*
     * История перестроений
     */


    if (
        !Array.isArray(
            context.replans
        )
    ) {

        context.replans = [];

    }








    context.replans.push({

        timestamp:

            new Date()
                .toISOString(),



        previousPlan:



            previousPlan?.intent ||
            null,



        newPlan:


            alternative.plan?.intent ||
            null,



        failure:

            alternative.failure || null


    });








    return true;

}









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
