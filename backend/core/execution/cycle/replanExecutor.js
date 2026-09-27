/*
 * =========================================================
 * JESSICA EXECUTION
 * REPLAN EXECUTOR v5
 * =========================================================
 *
 * Выполнение уже принятого Replan решения.
 *
 *
 * Flow:
 *
 * Replan Decision
 *        ↓
 * Alternative Plan
 *        ↓
 * Apply Alternative Plan
 *        ↓
 * Register Replan
 *        ↓
 * Reset Execution Pass
 *        ↓
 * Continue Execution
 *
 *
 * Ответственность:
 *
 * - запросить Alternative Plan;
 * - применить Alternative Plan;
 * - зарегистрировать успешный Replan;
 * - подготовить новый Execution Pass;
 * - зафиксировать lifecycle events в Trace.
 *
 *
 * НЕ:
 *
 * - принимает решение Replan;
 * - анализирует Failure;
 * - запускает Execution;
 * - хранит Replan history в Trace;
 * - изменяет счётчики напрямую.
 *
 * =========================================================
 */


import {
    createAlternativePlan,
    applyAlternativePlan
} from "../replanCoordinator.js";


import {
    resetExecutionAfterReplan,
    registerExecutionReplan
} from "../executionContext.js";


import {
    addTraceEvent
} from "../../trace/executionTrace.js";









/*
 * =========================================================
 * EXECUTE REPLAN
 * =========================================================
 */


export async function executeReplan(

    context,

    decision

) {


    /*
     * =====================================================
     * GUARD
     * =====================================================
     */


    if (

        !context ||

        !decision

    ) {


        return false;

    }









    /*
     * =====================================================
     * START
     * =====================================================
     */


    addTraceEvent(

        context.trace,

        "REPLAN_STARTED",

        {

            failure:

                decision.failure || null

        }

    );









    const previousPlan =

        context.plan || null;









    /*
     * =====================================================
     * CREATE ALTERNATIVE
     * =====================================================
     */


    let alternative;



    try {


        alternative =

            await createAlternativePlan(

                context,

                decision.failure

            );


    }

    catch(error){


        addTraceEvent(

            context.trace,

            "REPLAN_FAILED",

            {

                stage:

                    "create-alternative-plan",



                reason:

                    error?.message ||

                    "Alternative plan creation failed"


            }

        );



        return false;


    }









    /*
     * =====================================================
     * ALTERNATIVE FAILED
     * =====================================================
     */


    if (

        !alternative?.success ||

        !alternative.plan

    ) {


        addTraceEvent(

            context.trace,

            "REPLAN_FAILED",

            {

                stage:

                    "create-alternative-plan",



                reason:

                    alternative?.reason ||

                    "Alternative plan not created"


            }

        );



        return false;

    }









    /*
     * =====================================================
     * APPLY ALTERNATIVE
     * =====================================================
     */


    const applied =

        applyAlternativePlan(

            context,

            alternative

        );









    if (

        !applied

    ) {


        addTraceEvent(

            context.trace,

            "REPLAN_FAILED",

            {

                stage:

                    "apply-alternative-plan",



                reason:

                    "Alternative plan was not applied"


            }

        );



        return false;

    }









    /*
     * =====================================================
     * REGISTER REPLAN
     * =====================================================
     *
     * Context является единственным
     * источником истины для Replan history.
     *
     * Trace получит эту историю позже через:
     *
     * traceContextAdapter
     *      ↓
     * syncTraceReplans()
     *
     * =====================================================
     */


    registerExecutionReplan(

        context,

        {

            previousPlan,


            newPlan:

                context.plan,



            failure:

                decision.failure || null


        }

    );









    /*
     * =====================================================
     * RESET EXECUTION PASS
     * =====================================================
     *
     * Сбрасываем только локальные counters
     * текущего маршрута:
     *
     * attempt
     * retryCount
     *
     * replanCount НЕ сбрасывается.
     *
     * =====================================================
     */


    resetExecutionAfterReplan(

        context

    );









    /*
     * =====================================================
     * TRACE EVENT
     * =====================================================
     *
     * Здесь Trace хранит только lifecycle event.
     *
     * Сам Replan history находится в Context.
     *
     * =====================================================
     */


    addTraceEvent(

        context.trace,

        "REPLAN_COMPLETED",

        {

            replanCount:

                context.replanCount,



            previousPlan:



                previousPlan,



            newPlan:

                context.plan


        }

    );









    return true;

}
