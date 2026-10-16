import {
    buildTaskDecomposerInstructions
} from "./decomposer/decomposerPrompt.js";


import {
    validateDecomposition
} from "./decomposer/decompositionValidator.js";


import {
    isDecomposerAvailable,
    requestDecomposition
} from "../ai/decomposerClient.js";


/*
 * =========================================================
 * JESSICA TASK DECOMPOSER v3
 * =========================================================
 *
 * User Task
 *      +
 * Memory Context
 *          ↓
 * Decomposer AI Client
 *          ↓
 * Parse
 *          ↓
 * Structural Validator
 *          ↓
 * Canonical Subtasks
 *
 *
 * Главный принцип:
 *
 * Subtask != Tool Step.
 *
 *
 * AI transport находится в:
 *
 * backend/ai/decomposerClient.js
 *
 * =========================================================
 */


/*
 * =========================================================
 * CLEAN JSON
 * =========================================================
 */


function cleanJsonText(
    text
) {

    if(
        typeof text !== "string"
    ){

        return "";

    }


    let value =

        text
            .replace(
                /```json/gi,
                ""
            )
            .replace(
                /```/g,
                ""
            )
            .trim();


    const firstBrace =

        value.indexOf(
            "{"
        );


    const lastBrace =

        value.lastIndexOf(
            "}"
        );


    if(
        firstBrace !== -1
        &&
        lastBrace > firstBrace
    ){

        value =

            value.slice(

                firstBrace,

                lastBrace + 1

            );

    }


    return value.trim();

}


/*
 * =========================================================
 * NORMALIZE MEMORY
 * =========================================================
 */


function normalizeMemoryContext(
    memoryContext
) {

    if(
        !memoryContext
        ||
        !Array.isArray(
            memoryContext.skills
        )
    ){

        return [];

    }


    return memoryContext.skills

        .map(
            skill => ({

                id:

                    skill?.id

                    ||

                    skill?.skillId

                    ||

                    null,


                name:

                    skill?.name

                    ||

                    "",


                description:

                    skill?.description

                    ||

                    "",


                workflow:

                    Array.isArray(
                        skill?.workflow
                    )

                        ? skill.workflow

                        : [],


                constraints:

                    Array.isArray(
                        skill?.constraints
                    )

                        ? skill.constraints

                        : []

            })
        )

        .filter(
            skill =>
                skill.name
        )

        .slice(
            0,
            20
        );

}


/*
 * =========================================================
 * MEMORY CONTEXT
 * =========================================================
 */


function buildEffectiveMemoryContext(
    normalizedMemory
) {

    return {

        hasExperience:

            normalizedMemory.length > 0,

        skills:
            normalizedMemory

    };

}


/*
 * =========================================================
 * FALLBACK
 * =========================================================
 *
 * AI Decomposer unavailable / failed:
 *
 * не ломаем задачу.
 *
 * Вся исходная задача становится
 * одной самостоятельной подзадачей.
 *
 * =========================================================
 */


function createFallback(
    task,
    memoryContext
) {

    return {

        success:
            true,

        fallback:
            true,

        decomposition: {

            originalTask:
                task,

            isComplex:
                false,

            memoryContext,

            subtasks: [

                {

                    id:
                        1,

                    text:
                        task,

                    dependsOn:
                        [],

                    contributesToFinalAnswer:
                        true,

                    memoryHints:

                        normalizeMemoryContext(
                            memoryContext
                        )

                }

            ]

        }

    };

}


/*
 * =========================================================
 * FALLBACK WITH LOG
 * =========================================================
 */


function createLoggedFallback({

    task,

    memoryContext,

    reason

}) {

    console.warn(

        "Jessica Task Decomposer fallback:",

        reason

        ||

        "unknown reason"

    );


    return createFallback(

        task,

        memoryContext

    );

}


/*
 * =========================================================
 * PARSE RESPONSE
 * =========================================================
 */


function parseDecomposerResponse(
    response
) {

    const raw =

        response
            ?.output_text
            ?.trim();


    if(
        !raw
    ){

        return {

            success:
                false,

            reason:
                "AI Decomposer вернул пустой ответ"

        };

    }


    try {


        return {

            success:
                true,

            data:

                JSON.parse(

                    cleanJsonText(
                        raw
                    )

                )

        };


    }catch(error){


        return {

            success:
                false,

            reason:
                "AI Decomposer вернул некорректный JSON"

        };

    }

}


/*
 * =========================================================
 * DECOMPOSE TASK
 * =========================================================
 */


export async function decomposeTask({

    task,

    memoryContext = null

} = {}) {

    /*
     * =====================================================
     * INPUT
     * =====================================================
     */


    const normalizedTask =

        typeof task === "string"

            ? task.trim()

            : "";


    if(
        !normalizedTask
    ){

        return {

            success:
                false,

            text:
                "Не передана задача"

        };

    }


    /*
     * =====================================================
     * MEMORY
     * =====================================================
     */


    const normalizedMemory =

        normalizeMemoryContext(
            memoryContext
        );


    const effectiveMemoryContext =

        buildEffectiveMemoryContext(
            normalizedMemory
        );


    /*
     * =====================================================
     * NO AI
     * =====================================================
     */


    if(
        !isDecomposerAvailable()
    ){

        return createLoggedFallback({

            task:
                normalizedTask,

            memoryContext:
                effectiveMemoryContext,

            reason:
                "AI Decomposer недоступен"

        });

    }


    /*
     * =====================================================
     * AI REQUEST
     * =====================================================
     */


    let response;


    try {


        response =

            await requestDecomposition({

                instructions:

                    buildTaskDecomposerInstructions(),

                task:
                    normalizedTask,

                memory:
                    normalizedMemory

            });


    }catch(error){


        console.error(

            "Jessica Task Decomposer AI error:",

            {

                status:
                    error?.status || 0,

                code:
                    error?.code || null,

                message:
                    error?.message || String(error)

            }

        );


        /*
         * AI Retry уже исчерпан.
         *
         * Decomposer не должен ломать
         * весь пользовательский запрос.
         */


        return createFallback(

            normalizedTask,

            effectiveMemoryContext

        );

    }


    /*
     * =====================================================
     * PARSE
     * =====================================================
     */


    const parsed =

        parseDecomposerResponse(
            response
        );


    if(
        !parsed.success
    ){

        return createLoggedFallback({

            task:
                normalizedTask,

            memoryContext:
                effectiveMemoryContext,

            reason:
                parsed.reason

        });

    }


    /*
     * =====================================================
     * VALIDATE
     * =====================================================
     */


    const validated =

        validateDecomposition({

            data:
                parsed.data,

            originalTask:
                normalizedTask,

            memoryContext:
                effectiveMemoryContext

        });


    if(
        !validated.success
    ){

        return createLoggedFallback({

            task:
                normalizedTask,

            memoryContext:
                effectiveMemoryContext,

            reason:

                validated.text

                ||

                "Некорректная decomposition"

        });

    }


    /*
     * =====================================================
     * LOG
     * =====================================================
     */


    console.log(

        "Jessica Task Decomposition:",

        {

            subtasks:

                validated
                    .decomposition
                    .subtasks
                    .map(
                        subtask => ({

                            id:
                                subtask.id,

                            dependsOn:
                                subtask.dependsOn,

                            text:
                                subtask.text

                        })
                    )

        }

    );


    return validated;

}
