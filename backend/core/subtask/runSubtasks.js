import {
    executeSubtask
} from "./executeSubtask.js";


import {
    buildSubtaskSummary
} from "./subtaskSummary.js";


import {
    resolveDependencyState
} from "./subtaskDependencies.js";


/*
 * =========================================================
 * JESSICA RUN SUBTASKS v2
 * =========================================================
 *
 * Последовательный Dependency-aware
 * Subtask Runner.
 *
 *
 * Flow:
 *
 * decomposition
 *      ↓
 * subtask
 *      ↓
 * dependency check
 *      ↓
 *
 * satisfied
 *   → executeSubtask()
 *
 * failed prerequisite
 *   → BLOCKED
 *
 * =========================================================
 */


/*
 * =========================================================
 * FAILED RESULT
 * =========================================================
 */


function buildFailedSubtaskResult(
    subtask,
    error
) {

    return {

        id:
            subtask?.id ?? null,

        text:

            typeof subtask?.text === "string"

                ? subtask.text

                : "",

        success:
            false,

        status:
            "FAILED",

        stage:
            "subtask",

        result:
            "Непредвиденная ошибка выполнения подзадачи",

        error:

            error?.message

            ||

            "unknown error"

    };

}


/*
 * =========================================================
 * BLOCKED RESULT
 * =========================================================
 */


function buildBlockedSubtaskResult(
    subtask,
    dependencyState
) {

    return {

        id:
            subtask?.id ?? null,

        text:

            typeof subtask?.text === "string"

                ? subtask.text

                : "",

        success:
            false,

        status:
            "BLOCKED",

        stage:
            "dependency",

        result:

            "Подзадача не выполнена, потому что обязательная предыдущая подзадача не дала пригодного результата",

        dependencies:

            dependencyState.dependencies,

        blockedBy:

            dependencyState.blockedBy,

        missingDependencies:

            dependencyState.missing

    };

}


/*
 * =========================================================
 * RUN SUBTASKS
 * =========================================================
 */


export async function runSubtasks(
    decomposition
) {

    const subtasks =

        Array.isArray(
            decomposition?.subtasks
        )

            ? decomposition.subtasks

            : [];


    /*
     * =====================================================
     * EMPTY
     * =====================================================
     */


    if(
        subtasks.length === 0
    ){

        return buildSubtaskSummary(
            []
        );

    }


    const results = [];


    /*
     * =====================================================
     * EXECUTION
     * =====================================================
     */


    for(
        const subtask
        of subtasks
    ){

        console.log(

            "Jessica subtask:",

            {

                id:
                    subtask?.id,

                text:
                    subtask?.text,

                dependsOn:

                    subtask?.dependsOn

                    ??

                    subtask?.dependencies

                    ??

                    []

            }

        );


        /*
         * =================================================
         * DEPENDENCY CHECK
         * =================================================
         */


        const dependencyState =

            resolveDependencyState({

                subtask,

                results

            });


        if(
            !dependencyState.satisfied
        ){

            const blockedResult =

                buildBlockedSubtaskResult(

                    subtask,

                    dependencyState

                );


            console.warn(

                "Jessica subtask blocked:",

                {

                    id:
                        subtask?.id,

                    dependencies:
                        dependencyState.dependencies,

                    blockedBy:
                        dependencyState.blockedBy,

                    missing:
                        dependencyState.missing

                }

            );


            results.push(
                blockedResult
            );


            continue;

        }


        /*
         * =================================================
         * EXECUTE
         * =================================================
         */


        try {


            const result =

                await executeSubtask(
                    subtask
                );


            results.push(
                result
            );


        }catch(error){


            console.error(

                `Jessica subtask failed ${subtask?.id}:`,

                error

            );


            results.push(

                buildFailedSubtaskResult(

                    subtask,

                    error

                )

            );

        }

    }


    /*
     * =====================================================
     * SUMMARY
     * =====================================================
     */


    return buildSubtaskSummary(
        results
    );

}
