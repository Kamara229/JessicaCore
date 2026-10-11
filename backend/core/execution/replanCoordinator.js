/*
 * =========================================================
 * JESSICA REPLAN COORDINATOR v11
 * =========================================================
 *
 * Координатор перестроения Execution Plan.
 *
 *
 * Flow:
 *
 * Failure Decision
 *        ↓
 * Known Failed Routes
 *        ↓
 * Replanner
 *        ↓
 * Alternative Plan
 *        ↓
 * Diversity Policy
 *        ↓
 *
 * unique
 *   → Apply
 *
 * duplicate
 *   → reject
 *   → ask Replanner for another strategy
 *
 *
 * Ответственность:
 *
 * - подготовить запрос Replanner;
 * - не допустить повтор уже проваленного маршрута;
 * - получить реальный Alternative Plan;
 * - применить новый Plan;
 * - очистить результаты предыдущего маршрута.
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
    validateAlternativePlanDiversity
} from "./replan/replanDiversityPolicy.js";


/*
 * =========================================================
 * LIMITS
 * =========================================================
 */


export const MAX_REPLAN_COUNT =
    3;


/*
 * Количество попыток получить
 * действительно новую стратегию
 * внутри одного Replan.
 */


const MAX_ALTERNATIVE_GENERATION_ATTEMPTS =
    3;


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
 * DIVERSIFICATION FAILURE
 * =========================================================
 */


function buildDiversifiedFailure({

    failure,

    generationAttempt,

    rejectedFingerprints

}) {

    if(
        generationAttempt <= 1
    ){

        return failure;

    }


    return {

        ...failure,

        reason:

            [
                failure.reason,

                "",

                "Предыдущая предложенная альтернативная стратегия была отклонена, потому что повторяла уже испробованный Execution route.",

                "Не повторяй тот же набор инструментов с теми же аргументами и источниками.",

                "Измени способ получения или проверки результата.",

                `Отклонённых альтернатив в текущем Replan: ${rejectedFingerprints.length}.`

            ]
                .filter(Boolean)
                .join("\n"),

        details: {

            ...(
                failure?.details
                &&
                typeof failure.details === "object"
                &&
                !Array.isArray(
                    failure.details
                )

                    ? failure.details

                    : {}
            ),

            diversificationRequired:
                true,

            generationAttempt,

            rejectedPlanFingerprints: [
                ...rejectedFingerprints
            ]

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
     * KNOWN FAILED / TRIED ROUTES
     * =====================================================
     */


    const knownFingerprints =

        collectKnownPlanFingerprints({

            currentPlan:
                context?.plan || null,

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


    /*
     * =====================================================
     * GENERATE DIVERSE ALTERNATIVE
     * =====================================================
     */


    for(
        let generationAttempt = 1;

        generationAttempt
        <=
        MAX_ALTERNATIVE_GENERATION_ATTEMPTS;

        generationAttempt++
    ){

        const effectiveFailure =

            buildDiversifiedFailure({

                failure:
                    normalizedFailure,

                generationAttempt,

                rejectedFingerprints

            });


        let result;


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

                                required:

                                    generationAttempt > 1,

                                generationAttempt,

                                rejectedPlanFingerprints: [
                                    ...rejectedFingerprints
                                ]

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
         * DIVERSITY
         * =================================================
         */


        const diversity =

            validateAlternativePlanDiversity({

                candidatePlan:
                    result.plan,

                knownFingerprints

            });


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

                    rejectedCount:
                        rejectedFingerprints.length

                }

            };

        }


        /*
         * Duplicate route.
         */


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

            "Replanner не смог сформировать новый Execution route: все предложенные планы повторяли уже испробованные стратегии",

        rejectedFingerprints

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
     * Старые результаты принадлежат
     * предыдущему маршруту.
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
