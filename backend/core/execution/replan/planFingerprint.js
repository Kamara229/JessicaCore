/*
 * =========================================================
 * JESSICA EXECUTION PLAN FINGERPRINT v2
 * =========================================================
 *
 * Определяет идентичность Execution Plan
 * на уровне исполняемого маршрута.
 *
 *
 * В fingerprint входят:
 *
 * - evidence.mode;
 * - requiresTools;
 * - tool sequence;
 * - tool arguments;
 * - зависимости $from.
 *
 *
 * Нормализуются:
 *
 * - произвольные step IDs;
 * - whitespace;
 * - регистр search query;
 * - URL hostname / fragment.
 *
 *
 * НЕ входят:
 *
 * - reasoningSummary;
 * - metadata;
 * - generatedAt;
 * - plannerAttempt;
 * - Experience diagnostics.
 *
 * =========================================================
 */


import {
    createHash
} from "node:crypto";


/*
 * =========================================================
 * NORMALIZE TEXT
 * =========================================================
 */


function normalizeText(
    value
) {

    return String(
        value ?? ""
    )
        .trim();

}


/*
 * =========================================================
 * NORMALIZE STRING
 * =========================================================
 */


function normalizeStringValue(
    value
) {

    return normalizeText(
        value
    )
        .replace(
            /\s+/g,
            " "
        );

}


/*
 * =========================================================
 * NORMALIZE SEARCH QUERY
 * =========================================================
 *
 * Search query сравниваем без учёта:
 *
 * - регистра;
 * - лишних пробелов.
 *
 * Поэтому:
 *
 * PostgreSQL Latest Release
 *
 * и
 *
 * postgresql   latest release
 *
 * считаются одним query.
 *
 * =========================================================
 */


function normalizeSearchQuery(
    value
) {

    return normalizeStringValue(
        value
    )
        .toLowerCase();

}


/*
 * =========================================================
 * NORMALIZE URL
 * =========================================================
 */


function normalizeUrl(
    value
) {

    const text =
        normalizeStringValue(
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


        /*
         * Fragment не меняет содержимое
         * загружаемого HTTP resource.
         */


        url.hash =
            "";


        /*
         * Scheme и hostname
         * регистронезависимы.
         */


        url.protocol =
            url.protocol.toLowerCase();


        url.hostname =
            url.hostname.toLowerCase();


        return url.toString();


    }catch(error){


        /*
         * Если это не абсолютный URL,
         * не уничтожаем исходное значение.
         */


        return text;

    }

}


/*
 * =========================================================
 * STEP ID MAP
 * =========================================================
 *
 * AI может назвать одинаковые шаги:
 *
 * step_1
 * search
 * find_source
 *
 * Поэтому реальные IDs заменяются
 * на positional IDs.
 *
 * =========================================================
 */


function buildStepIdMap(
    steps
) {

    const map =
        new Map();


    const source =

        Array.isArray(
            steps
        )

            ? steps

            : [];


    source.forEach(
        (
            step,
            index
        ) => {

            const originalId =

                normalizeText(
                    step?.id
                );


            if(
                originalId
            ){

                map.set(

                    originalId,

                    `step_${index + 1}`

                );

            }

        }
    );


    return map;

}


/*
 * =========================================================
 * CANONICALIZE VALUE
 * =========================================================
 */


function canonicalizeValue(
    value,
    stepIdMap
) {

    /*
     * =====================================================
     * ARRAY
     * =====================================================
     */


    if(
        Array.isArray(
            value
        )
    ){

        return value.map(
            item =>
                canonicalizeValue(

                    item,

                    stepIdMap

                )
        );

    }


    /*
     * =====================================================
     * OBJECT
     * =====================================================
     */


    if(
        value
        &&
        typeof value === "object"
    ){

        const keys =

            Object.keys(
                value
            )
                .sort();


        const result =
            {};


        for(
            const key
            of keys
        ){

            let item =
                value[key];


            /*
             * Нормализуем dependency reference.
             */


            if(
                key === "$from"
            ){

                const reference =

                    normalizeText(
                        item
                    );


                item =

                    stepIdMap.get(
                        reference
                    )

                    ||

                    reference;

            }


            result[key] =

                canonicalizeValue(

                    item,

                    stepIdMap

                );

        }


        return result;

    }


    /*
     * =====================================================
     * STRING
     * =====================================================
     */


    if(
        typeof value === "string"
    ){

        return normalizeStringValue(
            value
        );

    }


    /*
     * =====================================================
     * PRIMITIVE
     * =====================================================
     */


    if(
        typeof value === "number"
        ||
        typeof value === "boolean"
        ||
        value === null
    ){

        return value;

    }


    return null;

}


/*
 * =========================================================
 * CANONICALIZE STEP ARGUMENTS
 * =========================================================
 */


function canonicalizeStepArguments(
    step,
    stepIdMap
) {

    const tool =

        normalizeText(
            step?.tool
        )
            .toLowerCase();


    const argumentsObject =

        canonicalizeValue(

            step?.arguments || {},

            stepIdMap

        );


    if(
        !argumentsObject
        ||
        typeof argumentsObject !== "object"
        ||
        Array.isArray(
            argumentsObject
        )
    ){

        return {};
    }


    /*
     * =====================================================
     * WEB SEARCH
     * =====================================================
     */


    if(
        tool === "web_search"
        &&
        typeof argumentsObject.query === "string"
    ){

        argumentsObject.query =

            normalizeSearchQuery(
                argumentsObject.query
            );

    }


    /*
     * =====================================================
     * WEB FETCH
     * =====================================================
     */


    if(
        tool === "web_fetch"
        &&
        typeof argumentsObject.url === "string"
    ){

        argumentsObject.url =

            normalizeUrl(
                argumentsObject.url
            );

    }


    return argumentsObject;

}


/*
 * =========================================================
 * CANONICAL ROUTE
 * =========================================================
 */


export function buildCanonicalExecutionPlan(
    plan
) {

    const steps =

        Array.isArray(
            plan?.steps
        )

            ? plan.steps

            : [];


    const stepIdMap =

        buildStepIdMap(
            steps
        );


    return {

        evidenceMode:

            normalizeText(
                plan?.evidence?.mode
            )
                .toLowerCase(),


        requiresTools:

            plan?.requiresTools === true,


        steps:

            steps.map(
                step => ({

                    tool:

                        normalizeText(
                            step?.tool
                        )
                            .toLowerCase(),


                    arguments:

                        canonicalizeStepArguments(

                            step,

                            stepIdMap

                        )

                })
            )

    };

}


/*
 * =========================================================
 * SIGNATURE
 * =========================================================
 */


export function buildPlanRouteSignature(
    plan
) {

    return JSON.stringify(

        buildCanonicalExecutionPlan(
            plan
        )

    );

}


/*
 * =========================================================
 * FINGERPRINT
 * =========================================================
 */


export function buildPlanFingerprint(
    plan
) {

    const signature =

        buildPlanRouteSignature(
            plan
        );


    return createHash(
        "sha256"
    )
        .update(
            signature
        )
        .digest(
            "hex"
        );

}


/*
 * =========================================================
 * EQUIVALENCE
 * =========================================================
 */


export function areExecutionPlansEquivalent(
    first,
    second
) {

    if(
        !first
        ||
        !second
    ){

        return false;

    }


    return (

        buildPlanFingerprint(
            first
        )

        ===

        buildPlanFingerprint(
            second
        )

    );

}
