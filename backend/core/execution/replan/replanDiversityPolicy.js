/*
 * =========================================================
 * JESSICA REPLAN DIVERSITY POLICY v1
 * =========================================================
 *
 * Alternative Plan
 *      ↓
 * Fingerprint
 *      ↓
 * Compare with failed / tried plans
 *      ↓
 *
 * NEW ROUTE
 *      → allowed
 *
 * SAME ROUTE
 *      → rejected
 *
 * =========================================================
 */


import {
    buildPlanFingerprint
} from "./planFingerprint.js";


/*
 * =========================================================
 * SAFE ARRAY
 * =========================================================
 */


function safeArray(
    value
) {

    return Array.isArray(value)

        ? value

        : [];

}


/*
 * =========================================================
 * COLLECT PLAN
 * =========================================================
 */


function addPlanFingerprint(
    target,
    plan
) {

    if(
        !plan
        ||
        typeof plan !== "object"
        ||
        Array.isArray(plan)
    ){

        return;

    }


    try {


        target.add(

            buildPlanFingerprint(
                plan
            )

        );


    }catch(error){


        /*
         * Fingerprint не должен ломать Execution.
         */

    }

}


/*
 * =========================================================
 * KNOWN PLANS
 * =========================================================
 */


export function collectKnownPlanFingerprints({

    currentPlan = null,

    initialPlan = null,

    replans = []

} = {}) {

    const fingerprints =
        new Set();


    addPlanFingerprint(

        fingerprints,

        currentPlan

    );


    addPlanFingerprint(

        fingerprints,

        initialPlan

    );


    for(
        const replan
        of safeArray(
            replans
        )
    ){

        addPlanFingerprint(

            fingerprints,

            replan?.previousPlan

        );


        addPlanFingerprint(

            fingerprints,

            replan?.newPlan

        );

    }


    return fingerprints;

}


/*
 * =========================================================
 * VALIDATE ALTERNATIVE
 * =========================================================
 */


export function validateAlternativePlanDiversity({

    candidatePlan,

    knownFingerprints

} = {}) {

    if(
        !candidatePlan
        ||
        typeof candidatePlan !== "object"
        ||
        Array.isArray(candidatePlan)
    ){

        return {

            success:
                false,

            reason:
                "Alternative Plan отсутствует",

            fingerprint:
                null

        };

    }


    const fingerprint =

        buildPlanFingerprint(
            candidatePlan
        );


    const known =

        knownFingerprints
        instanceof Set

            ? knownFingerprints

            : new Set();


    if(
        known.has(
            fingerprint
        )
    ){

        return {

            success:
                false,

            duplicate:
                true,

            reason:

                "Replanner вернул уже испробованный Execution route",

            fingerprint

        };

    }


    return {

        success:
            true,

        duplicate:
            false,

        reason:
            null,

        fingerprint

    };

}
