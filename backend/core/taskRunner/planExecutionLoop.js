/*
 * =========================================================
 * JESSICA TASK RUNNER
 * PLAN EXECUTION LOOP v3
 * =========================================================
 *
 * Plan
 *      ↓
 * Runtime Source Selection Context
 *      ↓
 * Steps
 *      ↓
 * executeStep()
 *
 *
 * Runtime context содержит только:
 *
 * - task;
 * - excludedSourceUrls;
 * - replanConstraints.
 *
 * Полный PlanningContext сюда
 * намеренно не передаётся.
 *
 * =========================================================
 */


import {
    executeStep
} from "./executeStep.js";


/*
 * =========================================================
 * SAFE
 * =========================================================
 */


function safeString(
    value
) {

    return String(
        value ?? ""
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
 * TASK TEXT
 * =========================================================
 */


function buildTaskText(
    plan,
    task
) {

    const explicitTask =

        safeString(
            task
        );


    if(
        explicitTask
    ){

        return explicitTask;

    }


    return [

        plan?.intent,

        plan?.reasoningSummary,

        plan?.evidence?.reason

    ]
        .map(
            safeString
        )
        .filter(Boolean)
        .join("\n");

}


/*
 * =========================================================
 * SELECTION CONTEXT
 * =========================================================
 */


function buildSelectionContext(
    plan,
    task
) {

    const runtime =

        plan
            ?.metadata
            ?.runtime;


    const runtimeObject =

        runtime
        &&
        typeof runtime === "object"
        &&
        !Array.isArray(
            runtime
        )

            ? runtime

            : {};


    const constraints =

        runtimeObject
            ?.replanConstraints;


    const constraintObject =

        constraints
        &&
        typeof constraints === "object"
        &&
        !Array.isArray(
            constraints
        )

            ? constraints

            : {};


    const excludedSourceUrls =

        uniqueStrings([

            ...safeArray(
                runtimeObject.triedSourceUrls
            ),

            ...safeArray(
                constraintObject.forbiddenSourceUrls
            )

        ]);


    return {

        task:

            buildTaskText(
                plan,
                task
            ),


        excludedSourceUrls,


        replanConstraints:

            constraintObject

    };

}


/*
 * =========================================================
 * EXECUTION LOOP
 * =========================================================
 */


export async function executePlanSteps({

    plan,

    task,

    results = []

}) {

    if(
        !Array.isArray(
            plan?.steps
        )
    ){

        return {

            success:
                false,

            stage:
                "runner",

            failureType:
                "invalid-steps",

            reason:
                "В плане отсутствуют шаги",

            results

        };

    }


    const selectionContext =

        buildSelectionContext(

            plan,

            task

        );


    /*
     * Runtime diagnostics.
     */


    if(
        selectionContext
            .excludedSourceUrls
            .length > 0
    ){

        console.log(

            "Jessica Source Exclusions:",

            JSON.stringify({

                count:

                    selectionContext
                        .excludedSourceUrls
                        .length,

                urls:

                    selectionContext
                        .excludedSourceUrls

            })

        );

    }


    for(
        let index = 0;

        index < plan.steps.length;

        index++
    ){

        const step =
            plan.steps[index];


        const result =

            await executeStep({

                step,

                index,

                results,

                selectionContext

            });


        if(
            result?.success !== true
        ){

            return result;

        }

    }


    return {

        success:
            true,

        results

    };

}
