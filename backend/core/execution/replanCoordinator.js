/*
 * =========================================================
 * JESSICA REPLAN COORDINATOR v10
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
 * Apply Alternative Plan
 *
 *
 * Ответственность:
 *
 * - подготовить запрос Replanner;
 * - получить новый Plan;
 * - применить новый Plan;
 * - очистить результаты предыдущего маршрута.
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


import {
    getExecutionCounters
} from "./executionContext.js";









/*
 * =========================================================
 * LIMIT
 * =========================================================
 */


export const MAX_REPLAN_COUNT = 3;









/*
 * =========================================================
 * SAFE STRING
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



        shouldRetry:

            failure?.shouldRetry === true,



        needsClarification:

            failure?.needsClarification === true,



        noVerifiedResult:

            failure?.noVerifiedResult === true,



        validation:

            failure?.validation ||

            null,



        source:

            failure?.source ||

            null,



        details:

            failure?.details ||

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


    const counters =

        getExecutionCounters(

            context

        );









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

                    counters?.attempt || 0

                ),



            retryCount:

                Number(

                    counters?.retryCount || 0

                ),



            replanCount:

                Number(

                    counters?.replanCount || 0

                )


        },



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


    if(

        !context

    ){


        return {


            success:false,


            reason:

                "Execution context is missing"


        };

    }









    const normalizedFailure =

        normalizeFailure(

            failure

        );









    const request =

        buildReplanRequest(

            context,

            normalizedFailure

        );









    /*
     * =====================================================
     * REPLAN LIMIT
     * =====================================================
     */


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









    /*
     * =====================================================
     * REPLANNER
     * =====================================================
     */


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









        /*
         * =================================================
         * INVALID REPLAN RESULT
         * =================================================
         */


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









        /*
         * =================================================
         * ALTERNATIVE
         * =================================================
         */


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
 * APPLY ALTERNATIVE PLAN
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









    /*
     * =====================================================
     * PLAN
     * =====================================================
     */


    context.plan =

        alternative.plan;









    /*
     * =====================================================
     * PLANNING CONTEXT
     * =====================================================
     */


    if(

        alternative.planningContext &&

        typeof alternative.planningContext === "object"

    ){


        context.planningContext =

            alternative.planningContext;

    }









    /*
     * =====================================================
     * CLEAR PREVIOUS ROUTE RESULTS
     * =====================================================
     *
     * Эти результаты принадлежат старому Plan.
     *
     * Новый маршрут обязан выполнить:
     *
     * TaskRunner
     *      ↓
     * Answer Composer
     *      ↓
     * Validator
     *
     * заново.
     *
     * =====================================================
     */


    context.runResult =

        null;



    context.answerResult =

        null;



    context.validationResult =

        null;



    context.terminalResult =

        null;









    return true;

}









/*
 * =========================================================
 * CAN REPLAN
 * =========================================================
 */


export function canReplan(

    context

) {


    if(

        !context

    ){


        return false;

    }









    const counters =

        getExecutionCounters(

            context

        );









    return (

        Number(

            counters?.replanCount || 0

        )

        <

        MAX_REPLAN_COUNT

    );

}
