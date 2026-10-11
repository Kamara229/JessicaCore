import OpenAI from "openai";


import {
    buildTaskDecomposerInstructions
} from "./decomposer/decomposerPrompt.js";


import {
    validateDecomposition
} from "./decomposer/decompositionValidator.js";


/*
 * =========================================================
 * JESSICA TASK DECOMPOSER v2
 * =========================================================
 *
 * User Task
 *      +
 * Memory Context
 *          ↓
 * AI Decomposer
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
 * Decomposer отвечает за смысловые цели.
 *
 * Planner отвечает за:
 *
 * web_search
 * web_fetch
 * tools
 * validation route
 *
 * =========================================================
 */


const groq =

    process.env.GROQ_API_KEY

        ? new OpenAI({

            apiKey:
                process.env.GROQ_API_KEY,

            baseURL:
                "https://api.groq.com/openai/v1"

        })

        : null;


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
 * FALLBACK
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
 * DECOMPOSE TASK
 * =========================================================
 */


export async function decomposeTask({

    task,

    memoryContext = null

} = {}) {

    const normalizedTask =

        typeof task === "string"

            ? task.trim()

            : "";


    /*
     * =====================================================
     * INPUT
     * =====================================================
     */


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


    const normalizedMemory =

        normalizeMemoryContext(
            memoryContext
        );


    const effectiveMemoryContext = {

        hasExperience:

            normalizedMemory.length > 0,

        skills:
            normalizedMemory

    };


    /*
     * =====================================================
     * NO AI
     * =====================================================
     */


    if(
        !groq
    ){

        return createFallback(

            normalizedTask,

            effectiveMemoryContext

        );

    }


    /*
     * =====================================================
     * AI DECOMPOSITION
     * =====================================================
     */


    try {


        const response =

            await groq.responses.create({

                model:
                    "openai/gpt-oss-20b",

                instructions:

                    buildTaskDecomposerInstructions(),

                input:

                    JSON.stringify({

                        task:
                            normalizedTask,

                        memory:
                            normalizedMemory

                    }),

                reasoning: {

                    effort:
                        "medium"

                }

            });


        const raw =

            response.output_text
                ?.trim();


        if(
            !raw
        ){

            return createFallback(

                normalizedTask,

                effectiveMemoryContext

            );

        }


        /*
         * =================================================
         * PARSE
         * =================================================
         */


        let parsed;


        try {


            parsed =

                JSON.parse(

                    cleanJsonText(
                        raw
                    )

                );


        }catch(error){


            console.warn(

                "Jessica Task Decomposer parse failed"

            );


            return createFallback(

                normalizedTask,

                effectiveMemoryContext

            );

        }


        /*
         * =================================================
         * VALIDATE
         * =================================================
         */


        const validated =

            validateDecomposition({

                data:
                    parsed,

                originalTask:
                    normalizedTask,

                memoryContext:
                    effectiveMemoryContext

            });


        if(
            !validated.success
        ){

            console.warn(

                "Jessica Task Decomposer validation failed:",

                validated.text

            );


            return createFallback(

                normalizedTask,

                effectiveMemoryContext

            );

        }


        /*
         * =================================================
         * LOG
         * =================================================
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


    }catch(error){


        console.error(

            "Jessica Task Decomposer error:",

            error

        );


        return createFallback(

            normalizedTask,

            effectiveMemoryContext

        );

    }

            }
