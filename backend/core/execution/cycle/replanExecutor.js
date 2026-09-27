/*
 * =========================================================
 * JESSICA EXECUTION
 * REPLAN EXECUTOR v3
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
 * Apply Context
 *        ↓
 * New Execution Route
 *
 *
 * НЕ:
 *
 * - принимает решение Replan;
 * - анализирует Failure;
 * - запускает Execution.
 *
 * =========================================================
 */


import {
    createAlternativePlan,
    applyAlternativePlan
} from "../replanCoordinator.js";


import {
    addTraceEvent,
    addTraceReplan
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


    if(

        !context ||

        !decision

    ){

        return false;

    }









    addTraceEvent(

        context.trace,

        "REPLAN_STARTED",

        {

            failure:

                decision.failure || null

        }

    );









    const previousPlan =

        context.plan;









    const alternative =

        await createAlternativePlan(

            context,

            decision.failure

        );









    if (

        !alternative?.success

    ) {


        addTraceEvent(

            context.trace,

            "REPLAN_FAILED",

            {

                reason:

                    alternative?.reason ||

                    "Alternative plan not created"

            }

        );


        return false;

    }









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

            "REPLAN_APPLY_FAILED"

        );


        return false;

    }









    /*
     * applyAlternativePlan()
     *
     * уже:
     *
     * - меняет Plan;
     * - очищает старый Result;
     * - сбрасывает Retry;
     * - регистрирует Replan.
     *
     * Здесь Execution Flow
     * только фиксирует Trace.
     */







    addTraceReplan(

        context.trace,

        {

            previousPlan,


            newPlan:

                context.plan,



            failure:

                decision.failure || null


        }

    );









    addTraceEvent(

        context.trace,

        "REPLAN_COMPLETED"

    );









    return true;

}
