import {
    createPlan
} from "../planner.js";


import {
    executePlanCycle
} from "../execution/executionCycle.js";


import {
    resolveExperience
} from "../../experience/experienceCore.js";


import {
    buildExecutionExperienceContext
} from "./subtaskExperienceContext.js";


/*
 * =========================================================
 * JESSICA EXECUTE SUBTASK v3
 * =========================================================
 *
 * Выполнение одной подзадачи.
 *
 *
 * Flow:
 *
 * Subtask
 *      ↓
 * Memory Context
 *      ↓
 * Experience Resolver
 *      ↓
 * Planning Context
 *      ↓
 * Planner
 *      ↓
 * Experience Adapter
 *      ↓
 * Execution Cycle
 *      ↓
 * Result
 *
 *
 * ВАЖНО:
 *
 * Experience передаётся:
 *
 * - Planner;
 * - Execution Context;
 * - Execution Trace;
 * - Learning.
 *
 * =========================================================
 */


/*
 * =========================================================
 * FAILED RESULT
 * =========================================================
 */


function buildFailedResult({

    subtaskId,

    taskText,

    stage,

    message,

    experience = null,

    plan = null,

    planningContext = null

}) {

    return {

        id:
            subtaskId,

        text:
            taskText,

        success:
            false,

        status:
            "FAILED",

        stage,

        result:
            message,

        experience,

        plan,

        planningContext

    };

}


/*
 * =========================================================
 * BUILD MEMORY EXPERIENCE
 * =========================================================
 */


function buildMemoryExperience(
    subtask
) {

    const hints =

        Array.isArray(
            subtask?.memoryHints
        )

            ? subtask.memoryHints

            : [];


    if(
        hints.length === 0
    ){

        return null;

    }


    return {

        source:
            "memory-context",

        skills:

            hints.map(

                item => ({

                    id:

                        item?.id

                        ||

                        null,

                    name:

                        item?.name

                        ||

                        "",

                    workflow:

                        Array.isArray(
                            item?.workflow
                        )

                            ? item.workflow

                            : [],

                    constraints:

                        Array.isArray(
                            item?.constraints
                        )

                            ? item.constraints

                            : []

                })

            )

    };

}


/*
 * =========================================================
 * MERGE EXPERIENCE
 * =========================================================
 */


function mergeExperienceContext(

    resolverExperience,

    memoryExperience

) {

    if(
        memoryExperience
    ){

        return {

            found:
                true,

            confidence:
                0.8,

            source:
                "memory-context",

            experience:
                memoryExperience,

            planningContext:
                null

        };

    }


    return resolverExperience || {

        found:
            false,

        confidence:
            0,

        source:
            "none",

        experience:
            null,

        planningContext:
            null

    };

}


/*
 * =========================================================
 * BUILD PLANNING CONTEXT
 * =========================================================
 */


function buildInitialPlanningContext({

    experienceResult,

    subtaskId,

    taskText

}) {

    const resolvedContext =

        experienceResult?.planningContext
        &&
        typeof experienceResult.planningContext === "object"
        &&
        !Array.isArray(
            experienceResult.planningContext
        )

            ? experienceResult.planningContext

            : null;


    if(
        resolvedContext
    ){

        return {

            ...resolvedContext,

            metadata: {

                ...(
                    resolvedContext.metadata
                    &&
                    typeof resolvedContext.metadata === "object"
                    &&
                    !Array.isArray(
                        resolvedContext.metadata
                    )

                        ? resolvedContext.metadata

                        : {}
                ),

                subtaskId

            }

        };

    }


    return {

        experience:
            experienceResult,

        metadata: {

            subtaskId,

            taskText,

            experienceSource:

                experienceResult?.source

                ||

                "none",

            experienceConfidence:

                Number(
                    experienceResult?.confidence || 0
                )

        }

    };

}


/*
 * =========================================================
 * EXECUTE SUBTASK
 * =========================================================
 */


export async function executeSubtask(
    subtask
) {

    const subtaskId =
        subtask?.id ?? null;


    const taskText =

        typeof subtask?.text === "string"

            ? subtask.text.trim()

            : "";


    /*
     * =====================================================
     * INPUT
     * =====================================================
     */


    if(
        !taskText
    ){

        return buildFailedResult({

            subtaskId,

            taskText,

            stage:
                "input",

            message:
                "Подзадача не содержит текста"

        });

    }


    /*
     * =====================================================
     * 1. EXPERIENCE RESOLUTION
     * =====================================================
     */


    let resolverResult =
        null;


    let memoryExperience =
        null;


    try {


        memoryExperience =

            buildMemoryExperience(
                subtask
            );


        /*
         * Если Decomposer уже передал
         * Memory Experience,
         * повторный Storage lookup
         * не выполняем.
         */


        if(
            !memoryExperience
        ){

            resolverResult =

                await resolveExperience(
                    taskText
                );

        }


    }catch(error){


        console.error(

            `Jessica Experience error ${subtaskId}:`,

            error

        );


        resolverResult = {

            found:
                false,

            confidence:
                0,

            source:
                "error",

            experience:
                null,

            planningContext:
                null

        };

    }


    const experienceResult =

        mergeExperienceContext(

            resolverResult,

            memoryExperience

        );


    /*
     * =====================================================
     * 2. EXECUTION EXPERIENCE
     * =====================================================
     *
     * Resolver format != Execution format.
     *
     * Здесь выполняется единственная
     * каноническая адаптация.
     *
     * =====================================================
     */


    const executionExperience =

        buildExecutionExperienceContext(
            experienceResult
        );


    /*
     * =====================================================
     * 3. PLANNING CONTEXT
     * =====================================================
     */


    const initialPlanningContext =

        buildInitialPlanningContext({

            experienceResult,

            subtaskId,

            taskText

        });


    console.log(

        `Jessica Subtask ${subtaskId} experience:`,

        {

            found:
                executionExperience.found,

            used:
                executionExperience.used,

            source:
                executionExperience.source,

            confidence:
                executionExperience.confidence,

            skills:

                executionExperience.skills

                    .map(
                        skill =>
                            skill?.id
                            ||
                            skill?.skillId
                            ||
                            skill?.name
                    )

                    .filter(Boolean)

        }

    );


    /*
     * =====================================================
     * 4. CREATE PLAN
     * =====================================================
     */


    let planResult;


    try {


        planResult =

            await createPlan(

                taskText,

                initialPlanningContext

            );


    }catch(error){


        console.error(

            `Jessica Planner error ${subtaskId}:`,

            error

        );


        return buildFailedResult({

            subtaskId,

            taskText,

            stage:
                "planner",

            message:
                "Ошибка создания плана",

            experience:
                executionExperience,

            planningContext:
                initialPlanningContext

        });

    }


    if(
        !planResult?.success
        ||
        !planResult?.plan
    ){

        return buildFailedResult({

            subtaskId,

            taskText,

            stage:
                "planner",

            message:

                planResult?.text

                ||

                "План не создан",

            experience:
                executionExperience,

            planningContext:
                initialPlanningContext

        });

    }


    /*
     * =====================================================
     * 5. EFFECTIVE PLANNING CONTEXT
     * =====================================================
     */


    const effectivePlanningContext =

        planResult.context
        &&
        typeof planResult.context === "object"
        &&
        !Array.isArray(
            planResult.context
        )

            ? planResult.context

            : initialPlanningContext;


    /*
     * =====================================================
     * 6. EXECUTION
     * =====================================================
     */


    let executionResult;


    try {


        /*
         * КРИТИЧЕСКО:
         *
         * четвёртый аргумент —
         * канонический Experience Context.
         *
         * Раньше здесь Experience терялся.
         */


        executionResult =

            await executePlanCycle(

                taskText,

                planResult.plan,

                effectivePlanningContext,

                executionExperience

            );


    }catch(error){


        console.error(

            `Jessica Execution Cycle error ${subtaskId}:`,

            error

        );


        return buildFailedResult({

            subtaskId,

            taskText,

            stage:
                "execution",

            message:
                "Ошибка цикла выполнения",

            experience:
                executionExperience,

            plan:
                planResult.plan,

            planningContext:
                effectivePlanningContext

        });

    }


    /*
     * =====================================================
     * 7. FINAL RESULT
     * =====================================================
     */


    return {

        id:
            subtaskId,

        text:
            taskText,

        experience: {

            found:
                executionExperience.found,

            used:
                executionExperience.used,

            source:
                executionExperience.source,

            confidence:
                executionExperience.confidence,

            skills: [

                ...executionExperience.skills

            ]

        },

        executionMeta: {

            experienceUsed:
                executionExperience.used,

            experienceFound:
                executionExperience.found,

            experienceSource:
                executionExperience.source,

            experienceConfidence:
                executionExperience.confidence,

            experienceSkills:
                executionExperience.skills.length,

            plannerUsed:
                true,

            executionStarted:
                true

        },

        plan:
            planResult.plan,

        planningContext:
            effectivePlanningContext,

        ...executionResult

    };

}
