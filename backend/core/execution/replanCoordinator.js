/*
 * =========================================================
 * JESSICA REPLAN COORDINATOR v9
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
 * - изменяет Execution counters;
 * - пишет Replan history;
 * - выполняет Execution.
 *
 * =========================================================
 */


import {
    replanTask
} from "../replanner.js";









export const MAX_REPLAN_COUNT = 3;









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

            "Execution plan failed",



        needsClarification:

            failure?.needsClarification === true,



        noVerifiedResult:

            failure?.noVerifiedResult === true,



        validation:

            failure?.validation || null


    };

}









function buildReplanRequest(

    context,

    failure

) {


    return {


        task:

            context?.task || "",



        previousPlan:

            context?.plan || null,



        failure,



        runResult:

            context?.runResult || null,



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









    if(

        request.executionState.replanCount

        >=

        MAX_REPLAN_COUNT

    ){

        return {


            success:false,


            reason:

                "Replan limit reached"


        };

    }









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









        if(

            !result ||

            result.success !== true ||

            !result.plan

        ){

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


    }

    catch(error){


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
 * APPLY PLAN
 * =========================================================
 */


export function applyAlternativePlan(

    context,

    alternative

) {


    if(

        !context ||

        !alternative?.success ||

        !alternative.plan

    ){

        return false;

    }









    context.plan =

        alternative.plan;









    if(

        alternative.planningContext

    ){

        context.planningContext =

            alternative.planningContext;

    }









    context.runResult = null;


    context.answerResult = null;


    context.validationResult = null;


    context.terminalResult = null;









    return true;

}









/*
 * =========================================================
 * CAN REPLAN
 * =========================================================
 */


export function canReplan(

    context

){

    return (

        Number(

            context?.replanCount || 0

        )

        <

        MAX_REPLAN_COUNT

    );

}
