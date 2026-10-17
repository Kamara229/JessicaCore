import {
    createPlan
} from "../planner.js";


import {
    tryCreateNativeSkillPlan
} from "../planner/nativeSkillPlanner.js";


import {
    tryCreateNativeCapabilityPlan
} from "../planner/nativeCapabilityPlanner.js";


import {
    getAIProviderAvailability
} from "../../ai/providerAvailability.js";


import {
    executePlanCycle
} from "../execution/executionCycle.js";


import {
    resolveExperience
} from "../../experience/experienceCore.js";


import {
    buildExecutionExperienceContext
} from "./subtaskExperienceContext.js";


import {
    buildSubtaskDependencyContext
} from "./subtaskDependencyContext.js";


/*
 * =========================================================
 * JESSICA EXECUTE SUBTASK v5
 * =========================================================
 *
 * Subtask
 *      ↓
 * Prerequisite Context
 *      ↓
 * Experience
 *      ↓
 * Planning Context
 *      ↓
 * Planner
 *      ↓
 * Execution Cycle
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
 * MEMORY EXPERIENCE
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

    /*
     * FULL Resolver Experience имеет приоритет.
     *
     * Memory hints от Decomposer компактные и
     * не содержат полного learning/maturity contract.
     * Они остаются fallback, если Storage/Resolver
     * не смог вернуть Skill.
     */


    if(
        resolverExperience?.found === true
        &&
        resolverExperience?.experience
    ){

        return resolverExperience;

    }


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
                resolverExperience?.planningContext
                ||
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
 * ARRAY
 * =========================================================
 */


function normalizeStringArray(
    value
) {

    return Array.isArray(value)

        ? value
            .map(
                item =>
                    String(
                        item || ""
                    )
                        .trim()
            )
            .filter(Boolean)

        : [];

}


/*
 * =========================================================
 * PLANNING CONTEXT
 * =========================================================
 */


function buildInitialPlanningContext({

    experienceResult,

    subtaskId,

    taskText,

    dependencyContext

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


    /*
     * Dependency data передаём как plannerHint.
     *
     * Это важно:
     *
     * prerequisite result помогает Planner,
     * но не имеет большего приоритета,
     * чем текущая пользовательская задача.
     */


    const dependencyHints =

        dependencyContext

            ? [
                dependencyContext
            ]

            : [];


    if(
        resolvedContext
    ){

        return {

            ...resolvedContext,

            plannerHints: [

                ...normalizeStringArray(
                    resolvedContext.plannerHints
                ),

                ...dependencyHints

            ],

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

                subtaskId,

                dependencyContextUsed:

                    Boolean(
                        dependencyContext
                    )

            }

        };

    }


    return {

        experience:
            experienceResult,

        plannerHints:
            dependencyHints,

        metadata: {

            subtaskId,

            taskText,

            dependencyContextUsed:

                Boolean(
                    dependencyContext
                ),

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
 * EXECUTE
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
     * 1. DEPENDENCY CONTEXT
     * =====================================================
     */


    const dependencyContext =

        buildSubtaskDependencyContext(
            subtask
        );


    /*
     * =====================================================
     * 2. EXPERIENCE RESOLUTION
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
         * Full Experience нужен не только AI Planner,
         * но и Native Skill Planner / Learning.
         *
         * Поэтому compact memoryHints больше
         * не отменяют обычный Experience Resolver.
         */


        resolverResult =

            await resolveExperience(
                taskText
            );


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
     * 3. EXECUTION EXPERIENCE
     * =====================================================
     */


    const executionExperience =

        buildExecutionExperienceContext(
            experienceResult
        );


    /*
     * =====================================================
     * 4. PLANNING CONTEXT
     * =====================================================
     */


    const initialPlanningContext =

        buildInitialPlanningContext({

            experienceResult,

            subtaskId,

            taskText,

            dependencyContext

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

                    .filter(Boolean),

            dependencyContext:

                Boolean(
                    dependencyContext
                )

        }

    );


    /*
     * =====================================================
     * 5. SKILL-FIRST PLAN
     * =====================================================
     *
     * Сначала Jessica пытается решить маршрут
     * своими native capabilities / READY Skills.
     *
     * Groq Planner вызывается только если
     * безопасного machine route нет.
     * =====================================================
     */


    let planResult;


    const nativePlan =

        tryCreateNativeSkillPlan({

            task:
                taskText,

            executionExperience,

            dependencyContext,

            planningContext:
                initialPlanningContext

        });


    if(
        nativePlan?.handled === true
        &&
        nativePlan?.plan
    ){

        planResult = {

            success:
                true,

            plan:
                nativePlan.plan,

            context:
                initialPlanningContext,

            source:
                nativePlan.source,

            native:
                true

        };


        console.log(

            `Jessica Native Skill Planner ${subtaskId}:`,

            {

                source:
                    nativePlan.source,

                intent:
                    nativePlan.plan.intent,

                steps:
                    nativePlan.plan.steps.length

            }

        );

    }
    else{


        /*
         * =================================================
         * PROVIDER COOLDOWN
         * =================================================
         *
         * Если Groq уже находится в cooldown,
         * даже не пытаемся отправлять новый AI request.
         * Сначала используем собственные capabilities.
         * =================================================
         */


        const providerState =

            getAIProviderAvailability();


        if(
            providerState.blocked === true
        ){

            const capabilityFallback =

                tryCreateNativeCapabilityPlan({

                    task:
                        taskText,

                    planningContext:
                        initialPlanningContext

                });


            if(
                capabilityFallback?.handled === true
                &&
                capabilityFallback?.plan
            ){

                planResult = {

                    success:
                        true,

                    plan:
                        capabilityFallback.plan,

                    context:
                        initialPlanningContext,

                    source:
                        capabilityFallback.source,

                    native:
                        true,

                    degraded:
                        true

                };


                console.log(

                    `Jessica Native Capability Planner ${subtaskId}:`,

                    {

                        source:
                            capabilityFallback.source,

                        reason:
                            "ai-provider-cooldown",

                        intent:
                            capabilityFallback.plan.intent,

                        steps:
                            capabilityFallback.plan.steps.length

                    }

                );

            }

        }


        /*
         * =================================================
         * AI PLANNER
         * =================================================
         */


        if(!planResult){

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


                /*
                 * AI недоступен — используем доступные
                 * native capabilities вместо немедленного
                 * падения всей задачи.
                 */


                const capabilityFallback =

                    tryCreateNativeCapabilityPlan({

                        task:
                            taskText,

                        planningContext:
                            initialPlanningContext

                    });


                if(
                    capabilityFallback?.handled === true
                    &&
                    capabilityFallback?.plan
                ){

                    planResult = {

                        success:
                            true,

                        plan:
                            capabilityFallback.plan,

                        context:
                            initialPlanningContext,

                        source:
                            capabilityFallback.source,

                        native:
                            true,

                        degraded:
                            true

                    };


                    console.log(

                        `Jessica Native Capability Planner ${subtaskId}:`,

                        {

                            source:
                                capabilityFallback.source,

                            reason:
                                "ai-planner-failed",

                            intent:
                                capabilityFallback.plan.intent,

                            steps:
                                capabilityFallback.plan.steps.length

                        }

                    );

                }
                else{

                    return buildFailedResult({

                        subtaskId,

                        taskText,

                        stage:
                            "planner",

                        message:
                            "Доступных локальных навыков недостаточно для построения безопасного плана",

                        experience:
                            executionExperience,

                        planningContext:
                            initialPlanningContext

                    });

                }

            }

        }

    }


    if(
        !planResult?.success
        ||
        !planResult?.plan
    ){

        /*
         * Semantic/technical AI Planner failure:
         * последняя попытка выполнить задачу
         * собственными возможностями Jessica.
         */


        const capabilityFallback =

            tryCreateNativeCapabilityPlan({

                task:
                    taskText,

                planningContext:
                    initialPlanningContext

            });


        if(
            capabilityFallback?.handled === true
            &&
            capabilityFallback?.plan
        ){

            planResult = {

                success:
                    true,

                plan:
                    capabilityFallback.plan,

                context:
                    initialPlanningContext,

                source:
                    capabilityFallback.source,

                native:
                    true,

                degraded:
                    true

            };


            console.log(

                `Jessica Native Capability Planner ${subtaskId}:`,

                {

                    source:
                        capabilityFallback.source,

                    reason:
                        "ai-plan-unusable",

                    intent:
                        capabilityFallback.plan.intent,

                    steps:
                        capabilityFallback.plan.steps.length

                }

            );

        }
        else{

            return buildFailedResult({

                subtaskId,

                taskText,

                stage:
                    "planner",

                message:

                    planResult?.text

                    ||

                    "Доступных локальных навыков недостаточно для безопасного выполнения задачи",

                experience:
                    executionExperience,

                planningContext:
                    initialPlanningContext

            });

        }

    }


    /*
     * =====================================================
     * 6. EFFECTIVE CONTEXT
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
     * 7. EFFECTIVE EXPERIENCE USAGE
     * =====================================================
     *
     * Native capability != Experience Skill.
     *
     * Нельзя усиливать случайно найденный Skill,
     * если фактически был использован только
     * встроенный machine capability.
     * =====================================================
     */


    const nativeExperienceUsed =

        planResult?.native === true

        &&

        String(
            planResult?.source || ""
        )
            .startsWith(
                "experience:"
            );


    const effectiveExecutionExperience =

        planResult?.native === true
        &&
        !nativeExperienceUsed

            ? {

                ...executionExperience,

                used:
                    false

            }

            : executionExperience;


    /*
     * =====================================================
     * 8. EXECUTION
     * =====================================================
     */


    let executionResult;


    try {


        executionResult =

            await executePlanCycle(

                taskText,

                planResult.plan,

                effectivePlanningContext,

                effectiveExecutionExperience

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
     * 9. RESULT
     * =====================================================
     */


    return {

        id:
            subtaskId,

        text:
            taskText,

        dependsOn:

            Array.isArray(
                subtask?.dependsOn
            )

                ? [
                    ...subtask.dependsOn
                ]

                : [],

        experience: {

            found:
                executionExperience.found,

            used:
                effectiveExecutionExperience.used,

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
                effectiveExecutionExperience.used,

            experienceFound:
                executionExperience.found,

            experienceSource:
                executionExperience.source,

            experienceConfidence:
                executionExperience.confidence,

            experienceSkills:
                executionExperience.skills.length,

            dependencyContextUsed:

                Boolean(
                    dependencyContext
                ),

            plannerUsed:
                planResult?.native !== true,

            nativePlannerUsed:
                planResult?.native === true,

            plannerSource:
                planResult?.source || "ai",

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
