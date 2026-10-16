/*
 * =========================================================
 * JESSICA REPLAN DIVERSITY POLICY v2
 * =========================================================
 *
 * Alternative Plan
 *      ↓
 * Execution Fingerprint
 *      ↓
 * Diversity Facts
 *      ↓
 * Compare with tried plans
 *
 *
 * NEW ROUTE
 *      → allowed
 *
 * SAME ROUTE
 *      → rejected
 *
 *
 * Дополнительно извлекаются:
 *
 * - search queries;
 * - static source URLs;
 * - tool sequence;
 * - route summary.
 *
 * Они нужны Replanner,
 * чтобы следующий generation attempt
 * получил конкретные запрещённые маршруты,
 * а не только SHA fingerprint.
 *
 * =========================================================
 */


import {
    buildPlanFingerprint
} from "./planFingerprint.js";


/*
 * =========================================================
 * SAFE
 * =========================================================
 */


function safeArray(
    value
) {

    return Array.isArray(
        value
    )

        ? value

        : [];

}


function safeString(
    value
) {

    return String(
        value ?? ""
    )
        .trim();

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
 * NORMALIZE QUERY
 * =========================================================
 */


function normalizeSearchQuery(
    value
) {

    return safeString(
        value
    )
        .replace(
            /\s+/g,
            " "
        )
        .toLowerCase();

}


/*
 * =========================================================
 * NORMALIZE URL
 * =========================================================
 */


function normalizeSourceUrl(
    value
) {

    const text =

        safeString(
            value
        );


    if(
        !text
    ){

        return "";

    }


    try {


        const url =
            new URL(
                text
            );


        url.hash =
            "";


        url.protocol =
            url.protocol.toLowerCase();


        url.hostname =
            url.hostname.toLowerCase();


        return url.toString();


    }catch(error){


        return text;

    }

}


/*
 * =========================================================
 * SEARCH QUERIES
 * =========================================================
 */


export function extractPlanSearchQueries(
    plan
) {

    return uniqueStrings(

        safeArray(
            plan?.steps
        )

            .filter(
                step =>
                    safeString(
                        step?.tool
                    )
                        .toLowerCase()
                    ===
                    "web_search"
            )

            .map(
                step =>
                    normalizeSearchQuery(
                        step?.arguments?.query
                    )
            )

            .filter(Boolean)

    );

}


/*
 * =========================================================
 * STATIC SOURCE URLS
 * =========================================================
 *
 * Извлекаем только URL,
 * непосредственно записанные в Plan.
 *
 * Dynamic $from URLs будут известны
 * только после TaskRunner и уже попадают
 * в Retry History.
 *
 * =========================================================
 */


export function extractPlanSourceUrls(
    plan
) {

    return uniqueStrings(

        safeArray(
            plan?.steps
        )

            .filter(
                step =>
                    safeString(
                        step?.tool
                    )
                        .toLowerCase()
                    ===
                    "web_fetch"
            )

            .map(
                step => {

                    const url =
                        step?.arguments?.url;


                    if(
                        typeof url !== "string"
                    ){

                        return "";

                    }


                    return normalizeSourceUrl(
                        url
                    );

                }
            )

            .filter(Boolean)

    );

}


/*
 * =========================================================
 * TOOL SEQUENCE
 * =========================================================
 */


export function extractPlanToolSequence(
    plan
) {

    return safeArray(
        plan?.steps
    )

        .map(
            step =>
                safeString(
                    step?.tool
                )
                    .toLowerCase()
        )

        .filter(Boolean);

}


/*
 * =========================================================
 * ROUTE SUMMARY
 * =========================================================
 */


export function buildPlanRouteSummary(
    plan
) {

    const evidenceMode =

        safeString(
            plan?.evidence?.mode
        )
            .toLowerCase();


    const tools =

        extractPlanToolSequence(
            plan
        );


    const queries =

        extractPlanSearchQueries(
            plan
        );


    const sourceUrls =

        extractPlanSourceUrls(
            plan
        );


    return [

        evidenceMode
            ? `evidence=${evidenceMode}`
            : null,

        tools.length > 0
            ? `tools=${tools.join(" -> ")}`
            : "tools=none",

        queries.length > 0
            ? `queries=${queries.join(" | ")}`
            : null,

        sourceUrls.length > 0
            ? `urls=${sourceUrls.join(" | ")}`
            : null

    ]
        .filter(Boolean)
        .join("; ");

}


/*
 * =========================================================
 * ADD PLAN FINGERPRINT
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
        Array.isArray(
            plan
        )
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
         * Fingerprint failure
         * не должен ломать Execution.
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
        Array.isArray(
            candidatePlan
        )
    ){

        return {

            success:
                false,

            duplicate:
                false,

            reason:
                "Alternative Plan отсутствует",

            fingerprint:
                null,

            searchQueries:
                [],

            sourceUrls:
                [],

            routeSummary:
                ""

        };

    }


    const fingerprint =

        buildPlanFingerprint(
            candidatePlan
        );


    const searchQueries =

        extractPlanSearchQueries(
            candidatePlan
        );


    const sourceUrls =

        extractPlanSourceUrls(
            candidatePlan
        );


    const routeSummary =

        buildPlanRouteSummary(
            candidatePlan
        );


    const known =

        knownFingerprints
        instanceof Set

            ? knownFingerprints

            : new Set();


    /*
     * =====================================================
     * EXACT EXECUTION DUPLICATE
     * =====================================================
     */


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

            fingerprint,

            searchQueries,

            sourceUrls,

            routeSummary

        };

    }


    /*
     * =====================================================
     * NEW ROUTE
     * =====================================================
     */


    return {

        success:
            true,

        duplicate:
            false,

        reason:
            null,

        fingerprint,

        searchQueries,

        sourceUrls,

        routeSummary

    };

}
