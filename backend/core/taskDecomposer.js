import OpenAI from "openai";


/*
 * =========================================================
 * JESSICA TASK DECOMPOSER
 * =========================================================
 *
 * Разделяет комплексную пользовательскую задачу
 * на независимые подзадачи.
 *
 * Пример:
 *
 * "Узнай время в Токио, погоду и курс иены"
 *
 * может стать:
 *
 * 1. Узнать текущее время в Токио
 * 2. Узнать текущую погоду в Токио
 * 3. Узнать актуальный курс иены
 *
 * ВАЖНО:
 *
 * Decomposer НЕ решает задачи.
 * НЕ выбирает инструменты.
 * НЕ отвечает пользователю.
 *
 * Его задача только определить структуру работы.
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
 * LIMITS
 * =========================================================
 */


const MAX_SUBTASKS =
    30;


/*
 * =========================================================
 * JSON CLEANUP
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
 * VALIDATION
 * =========================================================
 */


function validateDecomposition(
    data,
    originalTask
) {

    if (
        !data ||
        typeof data !== "object"
    ) {

        return {
            success: false,

            text:
                "Decomposer вернул некорректную структуру"
        };

    }


    if (
        !Array.isArray(
            data.subtasks
        )
    ) {

        return {
            success: false,

            text:
                "Decomposer не сформировал список подзадач"
        };

    }


    if (
        data.subtasks.length === 0
    ) {

        return {
            success: false,

            text:
                "Decomposer сформировал пустой список подзадач"
        };

    }


    if (
        data.subtasks.length >
        MAX_SUBTASKS
    ) {

        return {
            success: false,

            text:
                `Слишком много подзадач: ${data.subtasks.length}`
        };

    }


    const subtasks =
        [];


    for (
        let index = 0;
        index < data.subtasks.length;
        index++
    ) {

        const item =
            data.subtasks[index];


        if (
            !item ||
            typeof item !== "object"
        ) {

            return {
                success: false,

                text:
                    `Некорректная подзадача ${index + 1}`
            };

        }


        const text =
            typeof item.text === "string"
                ? item.text.trim()
                : "";


        if (!text) {

            return {
                success: false,

                text:
                    `Подзадача ${index + 1} не содержит текста`
            };

        }


        subtasks.push({

            id:
                index + 1,

            text,

            /*
             * Если результат этой подзадачи
             * понадобится для общей финальной сводки.
             */
            contributesToFinalAnswer:
                item.contributesToFinalAnswer !== false

        });

    }


    return {
        success: true,

        decomposition: {

            originalTask,

            isComplex:
                subtasks.length > 1,

            subtasks

        }

    };

}


/*
 * =========================================================
 * FALLBACK
 * =========================================================
 *
 * Если Decomposer временно недоступен,
 * не ломаем Jessica.
 *
 * Весь запрос становится одной подзадачей.
 */


function createSingleTaskFallback(
    task
) {

    return {

        success: true,

        fallback:
            true,

        decomposition: {

            originalTask:
                task,

            isComplex:
                false,

            subtasks: [
                {
                    id:
                        1,

                    text:
                        task,

                    contributesToFinalAnswer:
                        true
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


export async function decomposeTask(
    task
) {

    const normalizedTask =
        typeof task === "string"
            ? task.trim()
            : "";


    if (!normalizedTask) {

        return {
            success: false,

            text:
                "Не передана задача для декомпозиции"
        };

    }


    if (!groq) {

        return createSingleTaskFallback(
            normalizedTask
        );

    }


    try {

        const response =
            await groq.responses.create({

                model:
                    "openai/gpt-oss-20b",

                instructions:
                    (
                        "Ты — Task Decomposer системы Jessica Core. " +

                        "Ты НЕ решаешь пользовательскую задачу. " +
                        "Ты НЕ отвечаешь пользователю. " +
                        "Ты НЕ выбираешь инструменты. " +

                        "Твоя задача — определить, содержит ли запрос " +
                        "несколько самостоятельных действий или вопросов. " +

                        "Если запрос является одной цельной задачей, " +
                        "создай ровно одну подзадачу. " +

                        "Если пользователь явно или по смыслу просит выполнить " +
                        "несколько независимых действий, раздели их на подзадачи. " +

                        "Не дроби простую задачу на искусственные этапы. " +

                        "Например поиск информации, её анализ и формирование ответа " +
                        "могут оставаться одной подзадачей, если это одна цель. " +

                        "Но список из двадцати разных вопросов должен стать " +
                        "двадцатью отдельными подзадачами. " +

                        "Сохраняй важные условия исходного запроса внутри каждой " +
                        "подзадачи, чтобы её можно было выполнить независимо. " +

                        "Не теряй числа, даты, названия, ограничения и формулировки пользователя. " +

                        "Верни ТОЛЬКО JSON без markdown. " +

                        "Формат: " +

                        JSON.stringify({
                            subtasks: [
                                {
                                    text:
                                        "полная самостоятельная формулировка подзадачи",
                                    contributesToFinalAnswer:
                                        true
                                }
                            ]
                        })
                    ),

                input:
                    normalizedTask,

                reasoning: {
                    effort:
                        "medium"
                }

            });


        const raw =
            response.output_text
                ?.trim();


        if (!raw) {

            console.error(
                "Task Decomposer returned empty response"
            );


            return createSingleTaskFallback(
                normalizedTask
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

        } catch {

            console.error(
                "Task Decomposer invalid JSON:",
                raw
            );


            return createSingleTaskFallback(
                normalizedTask
            );

        }


        const validated =
            validateDecomposition(
                parsed,
                normalizedTask
            );


        if (
            !validated.success
        ) {

            console.error(
                "Task Decomposer validation failed:",
                validated.text
            );


            return createSingleTaskFallback(
                normalizedTask
            );

        }


        return validated;


    } catch (error) {

        console.error(
            "Task Decomposer error:",
            error
        );


        return createSingleTaskFallback(
            normalizedTask
        );

    }

}
