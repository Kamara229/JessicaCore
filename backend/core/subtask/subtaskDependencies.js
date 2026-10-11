/*
 * =========================================================
 * JESSICA SUBTASK DEPENDENCIES v2
 * =========================================================
 *
 * Управляет Runtime dependency contract.
 *
 *
 * Decomposer:
 *
 * dependsOn:[1]
 *
 *        ↓
 *
 * Runner:
 *
 * prerequisite success?
 *
 * YES → execute
 * NO  → BLOCKED
 *
 * =========================================================
 */


/*
 * =========================================================
 * ID
 * =========================================================
 */


function normalizeId(
    value
) {

    if(
        value === null
        ||
        value === undefined
    ){

        return "";

    }


    return String(
        value
    )
        .trim();

}


/*
 * =========================================================
 * DEPENDENCIES
 * =========================================================
 */


export function resolveSubtaskDependencies(
    subtask
) {

    const raw =

        subtask?.dependsOn

        ??

        subtask?.dependencies

        ??

        subtask?.depends_on

        ??

        [];


    const source =

        Array.isArray(raw)

            ? raw

            : [raw];


    const result = [];


    for(
        const value
        of source
    ){

        const id =

            normalizeId(
                value
            );


        if(
            id
            &&
            !result.includes(
                id
            )
        ){

            result.push(
                id
            );

        }

    }


    return result;

}


/*
 * =========================================================
 * OUTCOME
 * =========================================================
 */


function resolveOutcomeType(
    result
) {

    return String(

        result
            ?.validation
            ?.outcomeType

        ??

        result
            ?.validationResult
            ?.outcomeType

        ??

        result
            ?.result
            ?.validation
            ?.outcomeType

        ??

        result?.outcomeType

        ??

        ""

    )
        .trim()
        .toLowerCase();

}


/*
 * =========================================================
 * SATISFIED RESULT
 * =========================================================
 */


export function isSubtaskDependencySatisfied(
    result
) {

    if(
        !result
        ||
        typeof result !== "object"
    ){

        return false;

    }


    if(
        result.success === false
    ){

        return false;

    }


    const status =

        String(
            result.status || ""
        )
            .trim()
            .toUpperCase();


    if(
        [
            "FAILED",
            "BLOCKED",
            "NEEDS_CLARIFICATION",
            "NO_VERIFIED_RESULT"
        ]
            .includes(
                status
            )
    ){

        return false;

    }


    if(
        resolveOutcomeType(
            result
        )
        ===
        "no_verified_result"
    ){

        return false;

    }


    return (

        result.success === true

        ||

        status === "COMPLETED"

    );

}


/*
 * =========================================================
 * RESULT MAP
 * =========================================================
 */


function buildResultMap(
    results
) {

    const map =
        new Map();


    for(
        const result
        of (
            Array.isArray(results)

                ? results

                : []
        )
    ){

        const id =

            normalizeId(
                result?.id
            );


        if(
            id
        ){

            map.set(
                id,
                result
            );

        }

    }


    return map;

}


/*
 * =========================================================
 * DEPENDENCY STATE
 * =========================================================
 */


export function resolveDependencyState({

    subtask,

    results

} = {}) {

    const dependencies =

        resolveSubtaskDependencies(
            subtask
        );


    if(
        dependencies.length === 0
    ){

        return {

            required:
                false,

            satisfied:
                true,

            dependencies:
                [],

            blockedBy:
                [],

            missing:
                []

        };

    }


    const resultMap =

        buildResultMap(
            results
        );


    const blockedBy = [];

    const missing = [];


    for(
        const dependencyId
        of dependencies
    ){

        if(
            !resultMap.has(
                dependencyId
            )
        ){

            missing.push(
                dependencyId
            );

            continue;

        }


        if(
            !isSubtaskDependencySatisfied(

                resultMap.get(
                    dependencyId
                )

            )
        ){

            blockedBy.push(
                dependencyId
            );

        }

    }


    return {

        required:
            true,

        satisfied:

            blockedBy.length === 0
            &&
            missing.length === 0,

        dependencies,

        blockedBy,

        missing

    };

}


/*
 * =========================================================
 * COLLECT RESULTS
 * =========================================================
 */


export function collectDependencyResults({

    subtask,

    results

} = {}) {

    const dependencies =

        resolveSubtaskDependencies(
            subtask
        );


    if(
        dependencies.length === 0
    ){

        return [];

    }


    const resultMap =

        buildResultMap(
            results
        );


    return dependencies

        .map(
            dependencyId =>
                resultMap.get(
                    dependencyId
                )
        )

        .filter(Boolean);

}
