import OpenAI from "openai";


import {
    executeAIWithRetry
} from "../ai/aiRetry.js";


/*
 * =========================================================
 * JESSICA COMPLEX ANSWER COMPOSER v2
 * =========================================================
 *
 * Объединяет результаты нескольких подзадач
 * в единый пользовательский ответ.
 *
 *
 * v2:
 *
 * - использует общую AI Retry Policy;
 * - 429 / timeout / 5xx не приводят
 *   сразу к fallback;
 * - semantic содержание результатов
 *   никогда не пересчитывается.
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
 * FORMAT RESULTS
 * =========================================================
 */


function formatResults(
    results
) {

    if(
        !Array.isArray(results)
        ||
        results.length === 0
    ){

        return "Результаты отсутствуют.";

    }


    return results

        .map(
            item => {

                return (

                    `Подзадача ${item.id ?? "?"}\n`

                    +

                    `Статус: ${item.status}\n`

                    +

                    `Результат:\n${item.result || ""}`

                );

            }
        )

        .join(
            "\n\n---\n\n"
        );

}


/*
 * =========================================================
 * FALLBACK
 * =========================================================
 */


function fallbackAnswer(
    subtaskRunResult
) {

    const results =

        Array.isArray(
            subtaskRunResult?.results
        )

            ? subtaskRunResult.results

            : [];


    if(
        results.length === 0
    ){

        return "Нет результатов выполнения.";

    }


    return results

        .map(
            item => {

                if(
                    item.status === "COMPLETED"
                ){

                    return (

                        `${item.id}) ${item.result}`

                    );

                }


                if(
                    item.status === "NEEDS_CLARIFICATION"
                ){

                    return (

                        `${item.id}) Требуется уточнение: ${item.result}`

                    );

                }


                if(
                    item.status === "BLOCKED"
                ){

                    return (

                        `${item.id}) Не выполнено: зависимая подзадача была заблокирована. ${item.result || ""}`

                    );

                }


                return (

                    `${item.id}) Не выполнено: ${item.result || ""}`

                );

            }
        )

        .join(
            "\n\n"
        );

}


/*
 * =========================================================
 * COMPOSE
 * =========================================================
 */


export async function composeComplexAnswer(

    originalTask,

    decomposition,

    subtaskRunResult

) {

    const results =

        Array.isArray(
            subtaskRunResult?.results
        )

            ? subtaskRunResult.results

            : [];


    /*
     * =====================================================
     * SINGLE RESULT
     * =====================================================
     */


    if(
        results.length === 1
        &&
        results[0]?.status === "COMPLETED"
    ){

        return {

            text:
                results[0].result || "",

            source:
                "single-result"

        };

    }


    /*
     * =====================================================
     * NO AI
     * =====================================================
     */


    if(
        !groq
    ){

        return {

            text:

                fallbackAnswer(
                    subtaskRunResult
                ),

            source:
                "fallback"

        };

    }


    /*
     * =====================================================
     * AI
     * =====================================================
     */


    try {


        const response =

            await executeAIWithRetry(

                () =>
                    groq.responses.create({

                        model:
                            "openai/gpt-oss-20b",

                        instructions:

                            `
Ты являешься модулем сборки ответа Jessica Core.

Тебе переданы уже готовые результаты подзадач.

Твои правила:

1. Не решай задачи заново.
2. Не используй внешние знания.
3. Не добавляй отсутствующие данные.
4. Не исправляй URL, email, телефоны, даты и числа.
5. Не меняй смысл результатов.
6. COMPLETED показывай как выполненный результат.
7. NEEDS_CLARIFICATION показывай как требующий уточнения.
8. FAILED показывай как не выполненный пункт.
9. BLOCKED показывай как заблокированный зависимостью пункт.
10. Не скрывай ошибки.
11. Не показывай внутреннюю архитектуру Jessica.

Ответ должен быть понятным пользователю.
Язык ответа — язык пользователя.
`,

                        input:

                            `
Исходная задача:

${originalTask}


Результаты подзадач:

${formatResults(results)}


Статистика:

Всего:
${subtaskRunResult.total || 0}

Выполнено:
${subtaskRunResult.completed || 0}

Уточнение:
${subtaskRunResult.needsClarification || 0}

Ошибки:
${subtaskRunResult.failed || 0}
`

                    }),

                {

                    label:
                        "Jessica Complex Composer",

                    maxAttempts:
                        3

                }

            );


        const text =

            response.output_text
                ?.trim();


        if(
            !text
        ){

            return {

                text:

                    fallbackAnswer(
                        subtaskRunResult
                    ),

                source:
                    "fallback"

            };

        }


        return {

            text,

            source:
                "groq"

        };


    }catch(error){


        console.error(

            "Jessica Complex Composer final error:",

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
         * Очень важно:
         *
         * сбой Composer не уничтожает
         * уже готовые результаты Subtasks.
         */


        return {

            text:

                fallbackAnswer(
                    subtaskRunResult
                ),

            source:
                "fallback"

        };

    }

}
