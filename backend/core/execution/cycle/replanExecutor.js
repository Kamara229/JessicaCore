/*
 * =========================================================
 * JESSICA EXECUTION REPLAN EXECUTOR v6
 * =========================================================
 *
 * Выполнение уже принятого Replan решения.
 *
 *
 * Дополнительная гарантия v6:
 *
 * Alternative Plan не может быть
 * идентичен текущему failed route.
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


import {
    areExecutionPlansEquivalent,
    buildPlanFingerprint
} from "../replan/planFingerprint.js";


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


    if(
        !context
        ||
        !decision
    ){

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

                decision.failure

                ||

                null

        }

    );


    const previousPlan =

        context.plan

        ||

        null;


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


    }catch(error){


        addTraceEvent(

            context.trace,

            "REPLAN_FAILED",

            {

                stage:
                    "create-alternative-plan",

                reason:

                    error?.message

                    ||

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


    if(
        !alternative?.success
        ||
        !alternative.plan
    ){

        addTraceEvent(

            context.trace,

            "REPLAN_FAILED",

            {

                stage:
                    "create-alternative-plan",

                reason:

                    alternative?.reason

                    ||

                    "Alternative plan not created",

                rejectedFingerprints:

                    Array.isArray(
                        alternative?.rejectedFingerprints
                    )

                        ? alternative.rejectedFingerprints

                        : []

            }

        );


        return false;

    }


    /*
     * =====================================================
     * FINAL DUPLICATE GUARD
     * =====================================================
     */


    if(
        previousPlan
        &&
        areExecutionPlansEquivalent(

            previousPlan,

            alternative.plan

        )
    ){

        addTraceEvent(

            context.trace,

            "REPLAN_REJECTED",

            {

                stage:
                    "diversity-check",

                reason:

                    "Alternative Plan повторяет текущий failed Execution route",

                fingerprint:

                    buildPlanFingerprint(
                        alternative.plan
                    )

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


    if(
        !applied
    ){

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
     */


    registerExecutionReplan(

        context,

        {

            previousPlan,

            newPlan:
                context.plan,

            failure:

                decision.failure

                ||

                null,

            diversification:

                alternative.diversification

                ||

                null

        }

    );


    /*
     * =====================================================
     * RESET EXECUTION PASS
     * =====================================================
     */


    resetExecutionAfterReplan(
        context
    );


    /*
     * =====================================================
     * TRACE
     * =====================================================
     */


    addTraceEvent(

        context.trace,

        "REPLAN_COMPLETED",

        {

            replanCount:
                context.replanCount,

            previousPlan,

            newPlan:
                context.plan,

            diversification:

                alternative.diversification

                ||

                null

        }

    );


    return true;

}
