import {
    createPlan
} from "./planner.js";
import { getAIProviderAvailability } from "../ai/providerAvailability.js";


import {
    buildRetryContext
} from "./replanner/retryContext.js";


/*
 * =========================================================
 * JESSICA REPLANNER v2
 * =========================================================
 *
 * Failure
 *      ↓
 * Retry Context
 *      ↓
 * Planner
 *      ↓
 * Validated Alternative Plan
 *      ↓
 * Runtime Replan Metadata
 *
 * =========================================================
 */


/*
 * =========================================================
 * SAFE CONTEXT
 * =========================================================
 */


function isObject(
    value
) {

    return Boolean(

        value
        &&
        typeof value === "object"
        &&
        !Array.isArray(
            value
        )

    );

}


/*
 * =========================================================
 * LOG
 * =========================================================
 */


function logRetryContext(
    retryContext
) {

    const metadata =
        retryContext?.metadata || {};


    console.log(

        "Jessica Replanner context:",

        JSON.stringify({

            replanCount:
                metadata.replanCount || 0,


            retry:
                metadata.retry || null,


            previousAttempt:
                metadata.previousAttempt || null,


            triedSearchQueries:
                metadata.triedSearchQueries || [],


            triedSourceUrls:
                metadata.triedSourceUrls || [],


            replanConstraints:
                metadata.replanConstraints || null

        })

    );

}


/*
 * =========================================================
 * MERGE EFFECTIVE CONTEXT
 * =========================================================
 */


function mergeEffectiveContext(
    retryContext,
    plannerContext
) {

    if(
        !isObject(
            plannerContext
        )
    ){

        return retryContext;
    }


    return {

        ...retryContext,

        ...plannerContext,


        metadata: {

            ...(
                isObject(
                    retryContext?.metadata
                )

                    ? retryContext.metadata

                    : {}
            ),

            ...(
                isObject(
                    plannerContext?.metadata
                )

                    ? plannerContext.metadata

                    : {}
            )

        }

    };

}


/*
 * =========================================================
 * ATTACH MACHINE RUNTIME METADATA
 * =========================================================
 *
 * TaskRunner не должен знать весь PlanningContext.
 *
 * Поэтому в Plan переносим только минимум,
 * необходимый во время выполнения.
 *
 * Fingerprint metadata не учитывает.
 *
 * =========================================================
 */


function attachRuntimeMetadata(
    plan,
    context
) {

    const metadata =

        isObject(
            plan?.metadata
        )

            ? plan.metadata

            : {};


    const contextMetadata =

        isObject(
            context?.metadata
        )

            ? context.metadata

            : {};


    return {

        ...plan,


        metadata: {

            ...metadata,


            runtime: {

                replanConstraints:

                    contextMetadata.replanConstraints
                    ||
                    null,


                triedSearchQueries:

                    Array.isArray(
                        contextMetadata.triedSearchQueries
                    )

                        ? [
                            ...contextMetadata.triedSearchQueries
                        ]

                        : [],


                triedSourceUrls:

                    Array.isArray(
                        contextMetadata.triedSourceUrls
                    )

                        ? [
                            ...contextMetadata.triedSourceUrls
                        ]

                        : []

            }

        }

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

    // A known provider quota outage is not a reason to attempt
    // another AI plan. Stop immediately, preserving failed evidence.
    const provider = getAIProviderAvailability();
    if (provider.blocked) {
        console.log("Jessica Replanner skipped: AI quota cooldown", {
            reason: provider.reason,
            blockedUntil: provider.blockedUntil
        });
        return {
            success: false,
            reason: "AI Planner недоступен из-за действующего ограничения провайдера",
            providerCooldown: true
        };
    }

    /*
     * 1. Retry context.
     */


    const retryContext =

        buildRetryContext({

            planningContext,

            previousPlan,

            previousRunResult,

            failureResult

        });


    logRetryContext(
        retryContext
    );


    /*
     * 2. Planner.
     */


    try {


        const result =

            await createPlan(

                task,

                retryContext

            );


        if(
            !result?.success
            ||
            !result?.plan
        ){

            return {

                success:
                    false,

                reason:

                    result?.text
                    ||
                    "Planner не смог создать альтернативный маршрут"

            };

        }


        /*
         * Planner context не должен
         * потерять retry metadata.
         */


        const effectiveContext =

            mergeEffectiveContext(

                retryContext,

                result?.context

            );


        /*
         * Runtime-only metadata для TaskRunner.
         */


        const runtimePlan =

            attachRuntimeMetadata(

                result.plan,

                effectiveContext

            );


        console.log(

            "Jessica Replanner new plan:",

            JSON.stringify(
                runtimePlan
            )

        );


        return {

            success:
                true,

            plan:
                runtimePlan,

            context:
                effectiveContext

        };


    }catch(error){


        console.error(

            "Jessica Replanner error:",

            error

        );


        return {

            success:
                false,

            reason:

                error?.message
                ||
                "Ошибка Replanner"

        };

    }

}
