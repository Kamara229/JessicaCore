/*
 * =========================================================
 * JESSICA REPLAN COORDINATOR v12
 * =========================================================
 *
 * Failure
 *      ↓
 * Failed / Tried Routes
 *      ↓
 * Replanner
 *      ↓
 * Alternative Plan
 *      ↓
 * Diversity Policy
 *
 *
 * unique
 *      → Apply
 *
 * duplicate
 *      → extract query / URL / route
 *      → add hard constraints
 *      → ask Planner again
 *
 *
 * КРИТИЧЕСКО:
 *
 * SHA fingerprint используется машиной.
 *
 * Planner получает человекочитаемые:
 *
 * - forbidden search queries;
 * - forbidden source URLs;
 * - rejected route summaries.
 *
 * =========================================================
 */


import {
    replanTask
} from "../replanner.js";


import {
    getExecutionCounters,
    getExecutionReplans,
    getInitialPlan
} from "./executionContext.js";


import {
    collectKnownPlanFingerprints,
    extractPlanSearchQueries,
    extractPlanSourceUrls,
    buildPlanRouteSummary,
    validateAlternativePlanDiversity
} from "./replan/replanDiversityPolicy.js";


/*
 * =========================================================
 * LIMITS
 * =========================================================
 */


export const MAX_REPLAN_COUNT =
    3;


const MAX_ALTERNATIVE_GENERATION_ATTEMPTS =
    3;


/*
 * =========================================================
 * SAFE
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


function safeArray(
    value
) {

    return Array.isArray(
        value
    )

        ? value

        : [];

}


/*
 * =========================================================
 * UNIQUE
 * =========================================================
 */


function uniqueStrings(
    values
) {

    return [

        ...new Set(

            safeArray(
                values
            )

                .map(
                    value =>
                        safeString(
                            value
                        )
                )

                .filter(Boolean)

        )

    ];

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

            failure?.validation

            ||

            null,


        source:

            failure?.source

            ||

            null,


        details:

            failure?.details

            ||

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

            context?.task

            ||

            "",


        previousPlan:

            context?.plan

            ||

            null,


        failure,


        runResult:

            context?.runResult

            ||

            null,


        executionState: {

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

            context?.planningContext

            ||

            {}

    };

}


/*
 * =========================================================
 * REJECTED PLAN FACTS
 * =========================================================
 */


function collectRejectedPlanFacts(
    rejectedPlans
) {

    const plans =

        safeArray(
            rejectedPlans
        );


    const searchQueries =
        [];


    const sourceUrls =
        [];


    const routeSummaries =
        [];


    for(
        const plan
        of plans
    ){

        searchQueries.push(

            ...extractPlanSearchQueries(
                plan
            )

        );


        sourceUrls.push(

            ...extractPlanSourceUrls(
                plan
            )

        );


        const summary =

            buildPlanRouteSummary(
                plan
            );


        if(
            summary
        ){

            routeSummaries.push(
                summary
            );

        }

    }


    return {

        searchQueries:

            uniqueStrings(
                searchQueries
            ),


        sourceUrls:

            uniqueStrings(
                sourceUrls
            ),


        routeSummaries:

            uniqueStrings(
                routeSummaries
            )

    };

}


/*
 * =========================================================
 * DIVERSIFIED FAILURE
 * =========================================================
 */


function buildDiversifiedFailure({

    failure,

    generationAttempt,

    rejectedFingerprints,

    rejectedPlans

}) {

    const rejectedFacts =

        collectRejectedPlanFacts(
            rejectedPlans
        );


    const reasonParts = [

        failure.reason

    ];


    /*
     * Первый generation attempt уже является Replan.
     *
     * Поэтому diversificationRequired=true
     * с первой попытки.
     */


    if(
        generationAttempt > 1
    ){

        reasonParts.push(

            "",

            (
                "Предыдущая предложенная альтернативная стратегия " +
                "была отклонена, потому что повторяла уже " +
                "испробованный Execution route."
            ),

            (
                "Следующий Plan обязан отличаться " +
                "по содержательной стратегии."
            ),

            (
                `Отклонённых альтернатив в текущем Replan: ` +
                `${rejectedFingerprints.length}.`
            )

        );

    }


    const existingDetails =

        failure?.details
        &&
        typeof failure.details === "object"
        &&
        !Array.isArray(
            failure.details
        )

            ? failure.details

            : {};


    return {

        ...failure,


        reason:

            reasonParts
                .filter(Boolean)
                .join("\n"),


        details: {

            ...existingDetails,


            /*
             * HARD REPLAN CONTRACT
             */


            diversificationRequired:
                true,


            generationAttempt,


            rejectedPlanFingerprints: [

                ...rejectedFingerprints

            ],


            forbiddenSearchQueries:

                rejectedFacts.searchQueries,


            forbiddenSourceUrls:

                rejectedFacts.sourceUrls,


            rejectedRouteSummaries:

                rejectedFacts.routeSummaries

        }

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

    /*
     * =====================================================
     * CONTEXT
     * =====================================================
     */


    if(
        !context
    ){

        return {

            success:
                false,

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

            success:
                false,

            reason:
                "Replan limit reached"

        };

    }


    /*
     * =====================================================
     * KNOWN EXECUTION ROUTES
     * =====================================================
     */


    const knownFingerprints =

        collectKnownPlanFingerprints({

            currentPlan:

                context?.plan

                ||

                null,


            initialPlan:

                getInitialPlan(
                    context
                )

                ||

                null,


            replans:

                getExecutionReplans(
                    context
                )

                ||

                []

        });


    const rejectedFingerprints =
        [];


    const rejectedPlans =
        [];


    /*
     * =====================================================
     * GENERATE ALTERNATIVE
     * =====================================================
     */


    for(
        let generationAttempt = 1;

        generationAttempt
        <=
        MAX_ALTERNATIVE_GENERATION_ATTEMPTS;

        generationAttempt++
    ){

        /*
         * =================================================
         * FAILURE + HARD DIVERSIFICATION CONTEXT
         * =================================================
         */


        const effectiveFailure =

            buildDiversifiedFailure({

                failure:
                    normalizedFailure,

                generationAttempt,

                rejectedFingerprints,

                rejectedPlans

            });


        let result;


        /*
         * =================================================
         * REPLANNER
         * =================================================
         */


        try {


            result =

                await replanTask(

                    request.task,

                    request.previousPlan,

                    effectiveFailure,

                    request.runResult,

                    {

                        ...request.planningContext,


                        replanContext: {

                            previousPlan:
                                request.previousPlan,


                            failure:
                                effectiveFailure,


                            executionState:
                                request.executionState,


                            diversification: {

                                /*
                                 * Любой Replan обязан
                                 * изменить неудачную стратегию.
                                 */

                                required:
                                    true,


                                generationAttempt,


                                rejectedPlanFingerprints: [

                                    ...rejectedFingerprints

                                ],


                                rejectedPlans:

                                    collectRejectedPlanFacts(
                                        rejectedPlans
                                    )

                            }

                        }

                    }

                );


        }catch(error){


            return {

                success:
                    false,

                reason:

                    error?.message

                    ||

                    "Replanner error"

            };

        }


        /*
         * =================================================
         * INVALID RESULT
         * =================================================
         */


        if(
            !result
            ||
            result.success !== true
            ||
            !result.plan
        ){

            return {

                success:
                    false,

                reason:

                    result?.reason

                    ||

                    "Replanner did not create plan"

            };

        }


        /*
         * =================================================
         * DIVERSITY VALIDATION
         * =================================================
         */


        const diversity =

            validateAlternativePlanDiversity({

                candidatePlan:
                    result.plan,

                knownFingerprints

            });


        /*
         * =================================================
         * NEW ROUTE
         * =================================================
         */


        if(
            diversity.success
        ){

            return {

                success:
                    true,


                plan:
                    result.plan,


                failure:
                    effectiveFailure,


                planningContext:

                    result.context
                    &&
                    typeof result.context === "object"

                        ? result.context

                        : request.planningContext,


                diversification: {

                    generationAttempt,


                    fingerprint:
                        diversity.fingerprint,


                    routeSummary:
                        diversity.routeSummary,


                    rejectedCount:
                        rejectedFingerprints.length

                }

            };

        }


        /*
         * =================================================
         * DUPLICATE ROUTE
         * =================================================
         */


        rejectedPlans.push(
            result.plan
        );


        if(
            diversity.fingerprint
        ){

            rejectedFingerprints.push(
                diversity.fingerprint
            );


            knownFingerprints.add(
                diversity.fingerprint
            );

        }


        console.warn(

            "Jessica Replanner duplicate route rejected:",

            {

                generationAttempt,


                fingerprint:

                    diversity.fingerprint

                    ||

                    null,


                searchQueries:

                    diversity.searchQueries

                    ||

                    [],


                sourceUrls:

                    diversity.sourceUrls

                    ||

                    [],


                routeSummary:

                    diversity.routeSummary

                    ||

                    null,


                reason:

                    diversity.reason

                    ||

                    "duplicate route"

            }

        );

    }


    /*
     * =====================================================
     * NO REAL ALTERNATIVE
     * =====================================================
     */


    return {

        success:
            false,


        reason:

            (
                "Replanner не смог сформировать новый Execution route: " +
                "все предложенные планы повторяли уже испробованные стратегии"
            ),


        rejectedFingerprints,


        rejectedRoutes:

            collectRejectedPlanFacts(
                rejectedPlans
            )

    };

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
        !context
        ||
        !alternative?.success
        ||
        !alternative.plan
    ){

        return false;

    }


    context.plan =
        alternative.plan;


    if(
        alternative.planningContext
        &&
        typeof alternative.planningContext === "object"
    ){

        context.planningContext =
            alternative.planningContext;

    }


    /*
     * Результаты относятся
     * к предыдущему маршруту.
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
