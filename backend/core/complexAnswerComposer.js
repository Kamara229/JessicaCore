import OpenAI from "openai";


/*
 * =========================================================
 * JESSICA COMPLEX ANSWER COMPOSER
 * =========================================================
 *
 * Собирает результаты нескольких подзадач
 * в единый ответ пользователю.
 *
 * Важно:
 *
 * - успешные ответы сохраняются;
 * - ошибки не скрываются;
 * - уточнения показываются отдельно;
 * - Composer не должен выдумывать
 *   отсутствующие результаты.
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


function formatSubtaskResults(
    results
) {

    if (
        !Array.isArray(results) ||
        results.length === 0
    ) {

        return "Нет результатов подзадач.";

    }


    return results
        .map(
            item => {

                return JSON.stringify(
                    {
                        id:
                            item.id,

                        task:
                            item.text,

                        status:
                            item.status,

                        success:
                            item.success,

                        result:
                            item.result,

                        needsClarification:
                            item.needsClarification === true,

                        usedTools:
                            item.usedTools || []
                    },
                    null,
                    2
                );

            }
        )
        .join(
            "\n\n"
        );

}


/*
 * =========================================================
 * SIMPLE FALLBACK
 * =========================================================
 *
 * Если Groq недоступен,
 * Jessica всё равно должна показать
 * пользователю результаты.
 */


function buildFallbackAnswer(
    subtaskRunResult
) {

    const results =
        Array.isArray(
            subtaskRunResult?.results
        )
            ? subtaskRunResult.results
            : [];


    if (
        results.length === 0
    ) {

        return (
            "Не удалось получить результаты " +
            "по составной задаче."
        );

    }


    return results
        .map(
            item => {

                const number =
                    item.id ?? "?";


                if (
                    item.status ===
                    "COMPLETED"
                ) {

                    return (
                        `${number}) ${item.result}`
                    );

                }


                if (
                    item.status ===
                    "NEEDS_CLARIFICATION"
                ) {

                    return (
                        `${number}) Требуется уточнение: ` +
                        `${item.result}`
                    );

                }


                return (
                    `${number}) Не удалось выполнить: ` +
                    `${item.result}`
                );

            }
        )
        .join(
            "\n\n"
        );

}


/*
 * =========================================================
 * COMPOSE COMPLEX ANSWER
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
     * Если Decomposer создал одну задачу,
     * этот Composer всё равно может
     * корректно вернуть её результат.
     */
    if (
        results.length === 1
    ) {

        const only =
            results[0];


        if (
            only.status ===
            "COMPLETED"
        ) {

            return {

                success: true,

                text:
                    only.result,

                source:
                    "single-subtask"

            };

        }

    }


    if (!groq) {

        return {

            success: true,

            text:
                buildFallbackAnswer(
                    subtaskRunResult
                ),

            source:
                "fallback"

        };

    }


    try {

        const formattedResults =
            formatSubtaskResults(
                results
            );


        const response =
            await groq.responses.create({

                model:
                    "openai/gpt-oss-20b",

                instructions:
                    (
                        "Ты — Complex Answer Composer системы Jessica Core. " +

                        "Пользователь отправил составную задачу, " +
                        "которая была разбита на несколько независимых подзадач. " +

                        "Твоя задача — собрать результаты в один понятный итоговый ответ. " +

                        "НЕ решай подзадачи заново. " +
                        "НЕ придумывай отсутствующие данные. " +
                        "НЕ изменяй факты, числа, даты, время, URL и другие точные данные, " +
                        "которые уже содержатся в результатах. " +

                        "Если подзадача COMPLETED — покажи её результат. " +

                        "Если подзадача NEEDS_CLARIFICATION — прямо укажи, " +
                        "что для этого пункта требуется уточнение пользователя. " +

                        "Если подзадача FAILED — честно укажи, что её выполнить не удалось. " +

                        "Не скрывай неудачные пункты. " +

                        "Не позволяй ошибке одной подзадачи делать весь ответ неуспешным. " +

                        "Сохраняй исходную нумерацию подзадач, если она есть. " +

                        "Если вопросов много, отвечай структурированно и компактно. " +

                        "Не используй markdown-таблицы с символами |. " +

                        "Не показывай внутренние JSON, Planner, TaskRunner, Tool Registry " +
                        "или техническую архитектуру Jessica, " +
                        "если пользователь сам об этом не спрашивает. " +

                        "Отвечай на языке пользователя."
                    ),

                input:
                    (
                        `ИСХОДНАЯ КОМПЛЕКСНАЯ ЗАДАЧА:\n` +
                        `${originalTask}\n\n` +

                        `ДЕКОМПОЗИЦИЯ:\n` +
                        `${JSON.stringify(decomposition, null, 2)}\n\n` +

                        `РЕЗУЛЬТАТЫ ПОДЗАДАЧ:\n` +
                        `${formattedResults}\n\n` +

                        `СВОДКА:\n` +
                        `Всего: ${subtaskRunResult?.total || 0}\n` +
                        `Выполнено: ${subtaskRunResult?.completed || 0}\n` +
                        `Требуют уточнения: ${subtaskRunResult?.needsClarification || 0}\n` +
                        `Не выполнено: ${subtaskRunResult?.failed || 0}`
                    ),

                reasoning: {
                    effort:
                        "medium"
                }

            });


        const answer =
            response.output_text
                ?.trim();


        if (!answer) {

            return {

                success: true,

                text:
                    buildFallbackAnswer(
                        subtaskRunResult
                    ),

                source:
                    "fallback"

            };

        }


        return {

            success: true,

            text:
                answer,

            source:
                "groq"

        };


    } catch (error) {

        console.error(
            "Complex Answer Composer error:",
            error
        );


        return {

            success: true,

            text:
                buildFallbackAnswer(
                    subtaskRunResult
                ),

            source:
                "fallback"

        };

    }

}
