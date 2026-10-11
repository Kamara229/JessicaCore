/*
 * =========================================================
 * JESSICA EXECUTION PLAN FINGERPRINT v1
 * =========================================================
 *
 * Определяет семантическую идентичность
 * Execution Plan на уровне маршрута.
 *
 *
 * Используется для защиты от:
 *
 * Failed Plan
 *      ↓
 * Replan
 *      ↓
 * фактически тот же Failed Plan
 *
 *
 * В fingerprint входят:
 *
 * - evidence.mode;
 * - tool sequence;
 * - tool arguments;
 * - зависимости $from.
 *
 *
 * НЕ входят:
 *
 * - reasoningSummary;
 * - metadata;
 * - generatedAt;
 * - plannerAttempt;
 * - Experience diagnostics;
 * - step.id как произвольное имя.
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
 * STEP ID MAP
 * =========================================================
 *
 * AI может назвать одинаковые шаги:
 *
 * step_1 / search / find_source
 *
 * Поэтому реальные ID заменяем
 * на стабильные positional IDs.
 * =========================================================
 */


function buildStepIdMap(
    steps
) {

    const map =
        new Map();


    const source =

        Array.isArray(steps)

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
 * NORMALIZE STRING VALUE
 * =========================================================
 */


function normalizeStringValue(
    value
) {

    const text =

        normalizeText(
            value
        );


    /*
     * Стабилизируем whitespace,
     * но НЕ меняем регистр обычного текста:
     *
     * path / URL / query теоретически
     * могут быть case-sensitive.
     */


    return text.replace(
        /\s+/g,
        " "
    );

}


/*
 * =========================================================
 * CANONICALIZE
 * =========================================================
 */


function canonicalizeValue(
    value,
    stepIdMap
) {

    if(
        Array.isArray(value)
    ){

        return value.map(
            item =>
                canonicalizeValue(
                    item,
                    stepIdMap
                )
        );

    }


    if(
        value
        &&
        typeof value === "object"
    ){

        const keys =

            Object.keys(value)
                .sort();


        const result = {};


        for(
            const key
            of keys
        ){

            let item =

                value[key];


            /*
             * Нормализация $from.
             *
             * search_source и step_1
             * становятся одним positional reference.
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


    if(
        typeof value === "string"
    ){

        return normalizeStringValue(
            value
        );

    }


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

                        canonicalizeValue(

                            step?.arguments || {},

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
