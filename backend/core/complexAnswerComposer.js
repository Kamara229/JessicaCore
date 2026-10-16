/*
 * =========================================================
 * JESSICA COMPLEX ANSWER COMPOSER v3
 * =========================================================
 *
 * Объединяет готовые результаты нескольких
 * подзадач в единый пользовательский ответ.
 *
 *
 * Flow:
 *
 * Subtask Results
 *      ↓
 * Complex Composer AI
 *      ↓
 * Final Answer
 *
 *
 * Если AI недоступен:
 *
 * Subtask Results
 *      ↓
 * Deterministic Fallback
 *
 *
 * КРИТИЧЕСКО:
 *
 * Complex Composer:
 *
 * - не решает задачи заново;
 * - не выполняет tools;
 * - не меняет результаты subtasks;
 * - не создаёт AI client;
 * - не управляет AI retry.
 *
 *
 * AI transport:
 *
 * backend/ai/complexComposerClient.js
 *
 * =========================================================
 */


import {
    isComplexComposerAvailable,
    requestComplexComposerAnswer
} from "../ai/complexComposerClient.js";


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

                return [

                    `Подзадача ${item?.id ?? "?"}`,

                    `Статус: ${item?.status || "UNKNOWN"}`,

                    "Результат:",

                    String(
                        item?.result || ""
                    )

                ]
                    .join("\n");

            }
        )

        .join(
            "\n\n---\n\n"
        );

}


/*
 * =========================================================
 * FALLBACK ITEM
 * =========================================================
 */


function buildFallbackItem(
    item
) {

    const id =
        item?.id ?? "?";


    const status =

        String(
            item?.status || ""
        )
            .trim()
            .toUpperCase();


    const result =

        String(
            item?.result || ""
        )
            .trim();


    /*
     * =====================================================
     * COMPLETED
     * =====================================================
     */


    if(
        status === "COMPLETED"
    ){

        return (

            `${id}) ${result || "Выполнено."}`

        );

    }


    /*
     * =====================================================
     * NEEDS CLARIFICATION
     * =====================================================
     */


    if(
        status === "NEEDS_CLARIFICATION"
    ){

        return (

            `${id}) Требуется уточнение: `

            +

            (
                result

                ||

                "недостаточно данных."
            )

        );

    }


    /*
     * =====================================================
     * BLOCKED
     * =====================================================
     */


    if(
        status === "BLOCKED"
    ){

        return (

            `${id}) Не выполнено: зависимая подзадача была заблокирована.`

            +

            (
                result
                    ? ` ${result}`
                    : ""
            )

        );

    }


    /*
     * =====================================================
     * FAILED / OTHER
     * =====================================================
     */


    return (

        `${id}) Не выполнено: `

        +

        (
            result

            ||

            "задача завершилась с ошибкой."
        )

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
            buildFallbackItem
        )

        .join(
            "\n\n"
        );

}


/*
 * =========================================================
 * INSTRUCTIONS
 * =========================================================
 */


function buildComplexComposerInstructions()
{

    return `
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
`.trim();

}


/*
 * =========================================================
 * INPUT
 * =========================================================
 */


function buildComplexComposerInput({

    originalTask,

    subtaskRunResult,

    results

}) {

    return `
Исходная задача:

${String(originalTask || "").trim()}


Результаты подзадач:

${formatResults(results)}


Статистика:

Всего:
${Number(subtaskRunResult?.total || 0)}

Выполнено:
${Number(subtaskRunResult?.completed || 0)}

Уточнение:
${Number(subtaskRunResult?.needsClarification || 0)}

Ошибки:
${Number(subtaskRunResult?.failed || 0)}
`.trim();

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
     * SINGLE COMPLETED RESULT
     * =====================================================
     */


    if(
        results.length === 1
        &&
        results[0]?.status === "COMPLETED"
    ){

        return {

            text:

                String(
                    results[0]?.result || ""
                ),

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
        !isComplexComposerAvailable()
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
     * AI REQUEST
     * =====================================================
     */


    try {


        const response =

            await requestComplexComposerAnswer({

                instructions:

                    buildComplexComposerInstructions(),

                input:

                    buildComplexComposerInput({

                        originalTask,

                        subtaskRunResult,

                        results

                    })

            });


        const text =

            response
                ?.output_text
                ?.trim();


        /*
         * =================================================
         * EMPTY RESPONSE
         * =================================================
         */


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


        /*
         * =================================================
         * SUCCESS
         * =================================================
         */


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

                    error?.message

                    ||

                    String(error)

            }

        );


        /*
         * AI Retry уже исчерпан.
         *
         * Готовые результаты Subtasks
         * не уничтожаем.
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
