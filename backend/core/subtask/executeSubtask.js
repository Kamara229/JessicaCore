import {
    createPlan
} from "../planner.js";


import {
    executePlanCycle
} from "../execution/executionCycle.js";


import {
    resolveExperience
} from "../../experience/experienceCore.js";



/*
 * =========================================================
 * JESSICA EXECUTE SUBTASK v2
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
 * Execution Cycle
 *      ↓
 * Result
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


    if (
        hints.length === 0
    ) {

        return null;

    }


    return {

        source:
            "memory-context",

        skills:
            hints.map(

                item => ({

                    id:
                        item?.id || null,

                    name:
                        item?.name || "",

                    workflow:
                        item?.workflow || [],

                    constraints:
                        item?.constraints || []

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


    if (
        memoryExperience
    ) {


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
 *
 * Experience Core уже умеет строить полноценный
 * PlanningContext.
 *
 * Если он существует — используем его как основной
 * источник контекста Planner.
 *
 * Memory Context пока передаётся через experience,
 * поскольку он имеет отдельный формат.
 *
 * =========================================================
 */


function buildInitialPlanningContext({

    experienceResult,

    subtaskId,

    taskText

}) {


    const resolvedContext =

        experienceResult?.planningContext &&

        typeof experienceResult.planningContext === "object"

            ? experienceResult.planningContext

            : null;


    if (
        resolvedContext
    ) {


        return {

            ...resolvedContext,

            metadata: {

                ...(
                    resolvedContext.metadata &&
                    typeof resolvedContext.metadata === "object"

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
                experienceResult?.source || "none",

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


    if (
        !taskText
    ) {


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
     * 1. EXPERIENCE CONTEXT
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
         * Если Decomposer уже передал опыт,
         * дополнительный поиск не выполняется.
         *
         * Если опыта нет —
         * ищем в Experience Storage.
         */


        if (
            !memoryExperience
        ) {


            resolverResult =
                await resolveExperience(
                    taskText
                );

        }


    } catch(error) {


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
     * PLANNING CONTEXT
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
                experienceResult.found,

            source:
                experienceResult.source,

            confidence:
                experienceResult.confidence,

            skill:
                experienceResult
                    ?.planningContext
                    ?.experience
                    ?.name || null

        }

    );


    /*
     * =====================================================
     * 2. CREATE PLAN
     * =====================================================
     */


    let planResult;


    try {


        planResult =
            await createPlan(

                taskText,

                initialPlanningContext

            );


    } catch(error) {


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
                experienceResult,

            planningContext:
                initialPlanningContext

        });

    }


    if (
        !planResult?.success ||
        !planResult?.plan
    ) {


        return buildFailedResult({

            subtaskId,

            taskText,

            stage:
                "planner",

            message:
                planResult?.text ||
                "План не создан",

            experience:
                experienceResult,

            planningContext:
                initialPlanningContext

        });

    }


    /*
     * =====================================================
     * 3. EFFECTIVE CONTEXT
     * =====================================================
     */


    const effectivePlanningContext =

        planResult.context &&

        typeof planResult.context === "object"

            ? planResult.context

            : initialPlanningContext;


    /*
     * =====================================================
     * 4. EXECUTION
     * =====================================================
     */


    let executionResult;


    try {


        executionResult =
            await executePlanCycle(

                taskText,

                planResult.plan,

                effectivePlanningContext

            );


    } catch(error) {


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
                experienceResult,

            plan:
                planResult.plan,

            planningContext:
                effectivePlanningContext

        });

    }


    /*
     * =====================================================
     * 5. FINAL RESULT
     * =====================================================
     */


    return {

        id:
            subtaskId,

        text:
            taskText,

        experience: {

            found:
                experienceResult.found,

            source:
                experienceResult.source,

            confidence:
                experienceResult.confidence,

            skills:

                experienceResult
                    ?.experience
                    ?.skills || []

        },

        executionMeta: {

            experienceUsed:
                experienceResult.found,

            experienceSource:
                experienceResult.source,

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
