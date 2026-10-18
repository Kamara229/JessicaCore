import {
    executeSubtask
} from "./executeSubtask.js";


import {
    buildSubtaskSummary
} from "./subtaskSummary.js";


import {
    resolveDependencyState,
    collectDependencyResults
} from "./subtaskDependencies.js";


/*
 * =========================================================
 * JESSICA RUN SUBTASKS v3
 * =========================================================
 *
 * Последовательный Dependency-aware Runner.
 *
 *
 * Flow:
 *
 * subtask
 *    ↓
 * dependency state
 *    ↓
 *
 * satisfied
 *    ↓
 * attach prerequisite results
 *    ↓
 * executeSubtask
 *
 *
 * failed prerequisite
 *    ↓
 * BLOCKED
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

            "Подзадача заблокирована: необходимый результат предыдущей подзадачи отсутствует или не подтверждён",

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
 * RUN
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


    if(
        subtasks.length === 0
    ){

        return buildSubtaskSummary(
            []
        );

    }


    const results = [];


    for(
        const subtask
        of subtasks
    ){

        /*
         * =================================================
         * DEPENDENCY STATE
         * =================================================
         */


        const dependencyState =

            resolveDependencyState({

                subtask,

                results

            });


        console.log(

            "Jessica subtask:",

            {

                id:
                    subtask?.id,

                textLength: typeof subtask?.text === "string" ? subtask.text.length : 0,

                dependsOn:
                    dependencyState.dependencies

            }

        );


        /*
         * =================================================
         * BLOCKED
         * =================================================
         */


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
         * PREREQUISITE RESULTS
         * =================================================
         */


        const dependencyResults =

            collectDependencyResults({

                subtask,

                results

            });


        const executableSubtask = {

            ...subtask,

            dependencyResults

        };


        /*
         * =================================================
         * EXECUTE
         * =================================================
         */


        try {


            const result =

                await executeSubtask(
                    executableSubtask
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


    return buildSubtaskSummary(
        results
    );

}
