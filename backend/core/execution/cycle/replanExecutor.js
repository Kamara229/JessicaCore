/*
 * =========================================================
 * JESSICA EXECUTION
 * REPLAN EXECUTOR v2
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
 * Reset Execution State
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
    resetExecutionAfterReplan
} from "../executionContext.js";


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


    addTraceEvent(

        context.trace,

        "REPLAN_STARTED",

        {

            failure:

                decision.failure

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

            alternative

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
     * Новый Plan начинает
     * новый execution pass
     */


    resetExecutionAfterReplan(

        context

    );









    addTraceReplan(

        context.trace,

        {

            previousPlan,


            newPlan:

                context.plan,



            failure:

                decision.failure


        }

    );









    addTraceEvent(

        context.trace,

        "REPLAN_COMPLETED"

    );









    return true;

}
