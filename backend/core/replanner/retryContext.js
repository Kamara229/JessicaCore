import {
    buildRetryHistory
} from "./retryHistory.js";


import {
    buildRetryInstructions
} from "./retryInstructions.js";


/*
 * =========================================================
 * JESSICA REPLANNER
 * RETRY CONTEXT v3
 * =========================================================
 *
 * Кроме текстовых instructions теперь создаёт
 * машинный replanConstraints.
 *
 * =========================================================
 */


/*
 * =========================================================
 * SAFE
 * =========================================================
 */


function safeString(
    value
) {

    return typeof value === "string"
        ? value.trim()
        : "";

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


function uniqueStrings(
    values
) {

    return [

        ...new Set(

            safeArray(
                values
            )

                .map(
                    safeString
                )

                .filter(Boolean)

        )

    ];

}


/*
 * =========================================================
 * BASE CONTEXT
 * =========================================================
 */


function normalizeBaseContext(
    planningContext
) {

    if(
        !planningContext
        ||
        typeof planningContext !== "object"
        ||
        Array.isArray(
            planningContext
        )
    ){

        return {};
    }


    return {

        ...planningContext,


        metadata: {

            ...(
                planningContext.metadata
                &&
                typeof planningContext.metadata === "object"
                &&
                !Array.isArray(
                    planningContext.metadata
                )

                    ? planningContext.metadata

                    : {}
            )

        },


        instructions:

            Array.isArray(
                planningContext.instructions
            )

                ? [
                    ...planningContext.instructions
                ]

                : []

    };

}


/*
 * =========================================================
 * FAILURE DETAILS
 * =========================================================
 */


function getFailureDetails(
    failureResult
) {

    const details =
        failureResult?.details;


    if(
        !details
        ||
        typeof details !== "object"
        ||
        Array.isArray(
            details
        )
    ){

        return {};
    }


    return details;

}


/*
 * =========================================================
 * MACHINE REPLAN CONSTRAINTS
 * =========================================================
 */


function buildReplanConstraints({

    failureResult,

    retryHistory

}) {

    const details =

        getFailureDetails(
            failureResult
        );


    return {

        diversificationRequired:

            details?.diversificationRequired
            ===
            true,


        forbiddenSearchQueries:

            uniqueStrings([

                ...safeArray(
                    retryHistory?.triedSearchQueries
                ),

                ...safeArray(
                    details?.forbiddenSearchQueries
                )

            ]),


        forbiddenSourceUrls:

            uniqueStrings([

                ...safeArray(
                    retryHistory?.triedSourceUrls
                ),

                ...safeArray(
                    details?.forbiddenSourceUrls
                )

            ]),


        rejectedRouteSummaries:

            uniqueStrings(
                details?.rejectedRouteSummaries
            )

    };

}


/*
 * =========================================================
 * RETRY METADATA
 * =========================================================
 */


function buildRetryMetadata(
    failureResult
) {

    return {

        stage:

            safeString(
                failureResult?.stage
            ),


        failureType:

            safeString(
                failureResult?.failureType
            ),


        reason:

            safeString(

                failureResult?.reason
                ||
                failureResult?.text

            ),


        timestamp:

            new Date()
                .toISOString()

    };

}


/*
 * =========================================================
 * REPLAN COUNT
 * =========================================================
 */


function getNextReplanCount(
    planningContext
) {

    const current =

        Number(

            planningContext
                ?.metadata
                ?.replanCount

        );


    if(
        !Number.isFinite(
            current
        )
        ||
        current < 0
    ){

        return 1;

    }


    return (

        Math.floor(
            current
        )

        +

        1

    );

}


/*
 * =========================================================
 * BUILD
 * =========================================================
 */


export function buildRetryContext({

    planningContext = {},

    previousPlan = null,

    previousRunResult = null,

    failureResult = {}

} = {}) {

    const baseContext =

        normalizeBaseContext(
            planningContext
        );


    /*
     * History.
     */


    const retryHistory =

        buildRetryHistory({

            planningContext:
                baseContext,

            previousPlan,

            previousRunResult

        });


    /*
     * Machine constraints.
     */


    const replanConstraints =

        buildReplanConstraints({

            failureResult,

            retryHistory

        });


    /*
     * Human-readable Planner instructions.
     */


    const instructions =

        buildRetryInstructions({

            planningContext:
                baseContext,

            failureResult,

            retryHistory

        });


    const retry =

        buildRetryMetadata(
            failureResult
        );


    const replanCount =

        getNextReplanCount(
            baseContext
        );


    return {

        ...baseContext,


        instructions,


        metadata: {

            ...(baseContext.metadata || {}),


            retry,


            previousAttempt:

                retryHistory.previousAttempt,


            triedSearchQueries:

                retryHistory.triedSearchQueries,


            triedSourceUrls:

                retryHistory.triedSourceUrls,


            /*
             * КРИТИЧЕСКО:
             *
             * структурированные ограничения
             * для машинных policy layers.
             */


            replanConstraints,


            replanCount

        }

    };

}
