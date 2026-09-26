import OpenAI from "openai";



/*
 * =========================================================
 * JESSICA TASK DECOMPOSER
 * =========================================================
 *
 * Разделяет пользовательскую задачу
 * на независимые подзадачи.
 *
 *
 * Flow:
 *
 * User Task
 *      +
 * Memory Context
 *          ↓
 * Task Decomposer
 *          ↓
 * Subtasks
 *
 *
 * НЕ:
 *
 * - выполняет задачи;
 * - вызывает инструменты;
 * - создаёт ответы;
 * - изменяет память.
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






const MAX_SUBTASKS =
    30;







/*
 * =========================================================
 * CLEAN JSON
 * =========================================================
 */


function cleanJsonText(
    text
) {


    if (
        typeof text !== "string"
    ) {

        return "";

    }


    return text

        .replace(
            /```json/gi,
            ""
        )

        .replace(
            /```/g,
            ""
        )

        .trim();

}







/*
 * =========================================================
 * NORMALIZE MEMORY
 * =========================================================
 */


function normalizeMemoryContext(
    memoryContext
) {


    if (
        !memoryContext ||
        !Array.isArray(
            memoryContext.skills
        )
    ) {

        return [];

    }



    return memoryContext.skills

        .map(

            skill => ({

                name:
                    skill.name || "",


                description:
                    skill.description || "",


                workflow:
                    skill.workflow || [],


                constraints:
                    skill.constraints || []

            })

        )

        .filter(
            skill =>
                skill.name
        );


}








/*
 * =========================================================
 * VALIDATE
 * =========================================================
 */


function validateDecomposition(
    data,
    originalTask,
    memoryContext
) {


    if (
        !data ||
        typeof data !== "object"
    ) {

        return {

            success:
                false,


            text:
                "Некорректная структура"

        };

    }





    if (
        !Array.isArray(
            data.subtasks
        )
    ) {


        return {

            success:
                false,


            text:
                "Нет списка подзадач"

        };

    }





    if (
        data.subtasks.length === 0
    ) {


        return {

            success:
                false,


            text:
                "Пустой список задач"

        };

    }






    if (
        data.subtasks.length >
        MAX_SUBTASKS
    ) {


        return {

            success:
                false,


            text:
                "Слишком много подзадач"

        };

    }







    const subtasks =
        [];





    for (
        let i = 0;
        i < data.subtasks.length;
        i++
    ) {



        const item =
            data.subtasks[i];



        const text =
            typeof item?.text === "string"

                ? item.text.trim()

                : "";





        if (
            !text
        ) {


            return {

                success:
                    false,


                text:
                    `Пустая подзадача ${i + 1}`

            };

        }





        subtasks.push({

            id:
                i + 1,


            text,



            contributesToFinalAnswer:
                item.contributesToFinalAnswer !== false,



            memoryHints:
                Array.isArray(
                    item.memoryHints
                )

                    ? item.memoryHints

                    : []

        });


    }






    return {


        success:
            true,



        decomposition: {


            originalTask,


            isComplex:
                subtasks.length > 1,



            memoryContext,



            subtasks


        }


    };

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



            subtasks:[


                {


                    id:
                        1,



                    text:
                        task,



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





    if (
        !normalizedTask
    ) {


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





    if (
        !groq
    ) {


        return createFallback(

            normalizedTask,

            {

                hasExperience:
                    normalizedMemory.length > 0,


                skills:
                    normalizedMemory

            }

        );

    }







    try {


        const response =
            await groq.responses.create({

                model:
                    "openai/gpt-oss-20b",




                instructions:

                    `
Ты — Task Decomposer системы Jessica Core.

Твоя задача:
разделить пользовательский запрос
на самостоятельные исполнимые подзадачи.

У тебя есть Memory Context Jessica.

Используй его только как подсказку:
- не считай его всегда правильным;
- не добавляй факты из памяти без необходимости;
- учитывай прошлые workflow и ограничения.

Ты НЕ:
- решаешь задачу;
- отвечаешь пользователю;
- выбираешь инструменты.

Если задача простая:
создай одну подзадачу.

Если задача содержит несколько независимых целей:
раздели её.

Каждая подзадача должна быть самостоятельной.

Верни только JSON:

{
 "subtasks":[
   {
    "text":"описание подзадачи",
    "contributesToFinalAnswer":true,
    "memoryHints":[]
   }
 ]
}
`,





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





        if (
            !raw
        ) {


            return createFallback(

                normalizedTask,

                memoryContext

            );

        }






        let parsed;



        try {


            parsed =
                JSON.parse(
                    cleanJsonText(
                        raw
                    )
                );



        } catch(error) {


            return createFallback(

                normalizedTask,

                memoryContext

            );

        }







        const validated =
            validateDecomposition(

                parsed,

                normalizedTask,

                memoryContext

            );







        if (
            !validated.success
        ) {


            return createFallback(

                normalizedTask,

                memoryContext

            );

        }







        return validated;





    } catch(error) {


        console.error(

            "Jessica Task Decomposer error:",

            error

        );



        return createFallback(

            normalizedTask,

            memoryContext

        );

    }


}
