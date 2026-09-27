/*
 * =========================================================
 * JESSICA REPLAN COORDINATOR v7
 * =========================================================
 *
 * Координатор перестроения Execution Plan.
 *
 *
 * Flow:
 *
 * Failure Decision
 *        ↓
 * Replan Request
 *        ↓
 * Replanner
 *        ↓
 * Alternative Plan
 *        ↓
 * Apply Context
 *
 *
 * Ответственность:
 *
 * - подготовить запрос Replanner;
 * - получить новый Plan;
 * - заменить текущий маршрут.
 *
 *
 * НЕ:
 *
 * - анализирует Failure;
 * - решает нужен ли Replan;
 * - выполняет Tools;
 * - запускает Execution;
 * - изменяет Learning.
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


export const MAX_REPLAN_COUNT = 3;









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
 * FAILURE NORMALIZE
 * =========================================================
 */


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
            "Execution plan failed",



        validation:

            failure?.validation ||
            null


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



        runResult:

            context?.runResult ||
            null,



        executionState:

        {


            attempt:

                Number(
                    context?.attempt || 0
                ),



            retryCount:

                Number(
                    context?.retryCount || 0
                ),



            replanCount:

                Number(
                    context?.replanCount || 0
                )


        },



        planningContext:

            context?.planningContext || {}

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

        request.executionState.replanCount

        >=

        MAX_REPLAN_COUNT

    ) {


        return {


            success:false,


            reason:
                "Replan limit reached"


        };


    }









    console.log(

        "Jessica Replan Coordinator:",

        {

            failureType:

                normalizedFailure.failureType,


            category:

                normalizedFailure.category,


            attempt:

                request.executionState.attempt,


            replanCount:

                request.executionState.replanCount


        }

    );









    try {


        const result =

            await replanTask(

                request.task,

                request.previousPlan,

                normalizedFailure,

                request.runResult,

                {


                    ...request.planningContext,



                    replanContext:

                    {


                        previousPlan:

                            request.previousPlan,



                        failure:

                            normalizedFailure,



                        executionState:

                            request.executionState


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

                    "Replanner did not create plan"


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



    } catch(error){



        console.error(

            "Jessica Replan error:",

            error

        );



        return {


            success:false,


            reason:

                error?.message ||

                "Replanner error"


        };


    }


}









/*
 * =========================================================
 * APPLY ALTERNATIVE PLAN
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









    const previousPlan =

        context.plan;









    /*
     * Replace route
     */


    context.plan =

        alternative.plan;








    if(

        alternative.planningContext

    ){


        context.planningContext =

            alternative.planningContext;


    }









    /*
     * Clear previous execution data
     */


    context.runResult =
        null;


    context.answerResult =
        null;


    context.validationResult =
        null;


    context.terminalResult =
        null;









    /*
     * Reset retry cycle
     */


    context.attempt =
        0;


    context.retryCount =
        0;









    /*
     * Register replan
     */


    context.replanCount =

        Number(
            context.replanCount || 0
        )
        +
        1;









    if(

        !Array.isArray(
            context.replanHistory
        )

    ){

        context.replanHistory = [];

    }








    context.replanHistory.push({

        timestamp:

            new Date()
            .toISOString(),



        previousPlan:

            previousPlan || null,



        newPlan:

            alternative.plan,



        failure:

            alternative.failure || null


    });








    return true;


}









/*
 * =========================================================
 * CHECK POSSIBILITY
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
