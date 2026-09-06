import OpenAI from "openai";

import {
    executeAIWithRetry
} from "../ai/aiRetry.js";


/*
 * =========================================================
 * JESSICA SOURCE SELECTOR
 * =========================================================
 *
 * Выбирает наиболее подходящий источник
 * из результатов web_search.
 *
 * Source Selector имеет право:
 *
 * 1. выбрать подходящий источник;
 * 2. отклонить ВСЮ поисковую выдачу,
 *    если ни один источник не удовлетворяет задаче.
 *
 * Это важно:
 *
 * плохой источник НЕ должен выбираться
 * только потому, что других нет.
 */


/*
 * =========================================================
 * AI CLIENT
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


const SOURCE_SELECTOR_MODEL =
    "openai/gpt-oss-20b";


/*
 * =========================================================
 * CLEAN JSON
 * =========================================================
 */


function cleanJsonText(
    text
) {

    let value =
        String(text || "")
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
        value.indexOf("{");


    const lastBrace =
        value.lastIndexOf("}");


    if (
        firstBrace !== -1 &&
        lastBrace > firstBrace
    ) {

        value =
            value.slice(
                firstBrace,
                lastBrace + 1
            );

    }


    return value;

}


/*
 * =========================================================
 * NORMALIZE RESULTS
 * =========================================================
 */


function normalizeResults(
    results
) {

    if (
        !Array.isArray(
            results
        )
    ) {

        return [];

    }


    return results
        .map(
            (
                item,
                index
            ) => ({
                index,

                title:
                    String(
                        item?.title || ""
                    ).trim(),

                url:
                    String(
                        item?.url || ""
                    ).trim(),

                snippet:
                    String(
                        item?.snippet || ""
                    ).trim()
            })
        )
        .filter(
            item =>
                item.url
        );

}


/*
 * =========================================================
 * NO SUITABLE SOURCE
 * =========================================================
 */


function noSuitableSource(
    reason
) {

    return {
        success:
            false,

        noSuitableSource:
            true,

        index:
            null,

        result:
            null,

        reason:
            String(
                reason ||
                "Подходящий источник не найден"
            ).trim()
    };

}


/*
 * =========================================================
 * AI REQUEST
 * =========================================================
 */


async function requestSelection(
    task,
    results
) {

    return await executeAIWithRetry(
        async () => {

            return await groq
                .chat
                .completions
                .create({

                    model:
                        SOURCE_SELECTOR_MODEL,

                    temperature:
                        0,

                    messages: [
                        {
                            role:
                                "system",

                            content: [
                                "Ты Source Selector системы Jessica Core.",

                                "",
                                "Ты НЕ отвечаешь пользователю.",
                                "Ты НЕ выполняешь поиск.",
                                "Ты НЕ придумываешь URL.",

                                "",
                                "Твоя задача — оценить переданные результаты поиска.",

                                "",
                                "Есть только два допустимых решения:",

                                "",
                                "1. SELECT",
                                "Выбрать один результат, если он действительно удовлетворяет требованиям задачи.",

                                "",
                                "2. REJECT",
                                "Отклонить всю выдачу, если подходящего результата среди переданных нет.",

                                "",
                                "КРИТИЧЕСКОЕ ПРАВИЛО:",
                                "никогда не выбирай неподходящий источник только потому, что он лучший среди плохих вариантов.",

                                "",
                                "Если задача требует официальный источник,",
                                "выбранный результат должен действительно выглядеть официальным источником нужной организации, ведомства, компании, проекта или автора.",

                                "",
                                "Если задача требует первоисточник,",
                                "не заменяй его пересказом, агрегатором, случайным PDF, форумом или сторонним сайтом.",

                                "",
                                "Если нужен конкретный документ или страница,",
                                "результат должен с высокой вероятностью содержать именно требуемый материал.",

                                "",
                                "Если требования задачи нельзя подтвердить по title, URL и snippet,",
                                "предпочитай REJECT вместо рискованного выбора.",

                                "",
                                "Количество результатов не имеет значения.",
                                "Даже если результат только один, его можно отклонить.",

                                "",
                                "Для SELECT верни JSON:",
                                JSON.stringify({
                                    decision:
                                        "select",

                                    index:
                                        0,

                                    reason:
                                        "почему источник соответствует задаче"
                                }),

                                "",
                                "Для REJECT верни JSON:",
                                JSON.stringify({
                                    decision:
                                        "reject",

                                    index:
                                        null,

                                    reason:
                                        "почему ни один источник не подходит"
                                }),

                                "",
                                "Верни только JSON."
                            ].join(
                                "\n"
                            )
                        },

                        {
                            role:
                                "user",

                            content: [
                                "ЗАДАЧА:",
                                String(
                                    task || ""
                                ).trim(),

                                "",
                                "РЕЗУЛЬТАТЫ ПОИСКА:",
                                JSON.stringify(
                                    results,
                                    null,
                                    2
                                )
                            ].join(
                                "\n"
                            )
                        }
                    ]

                });

        },
        {
            label:
                "Source Selector"
        }
    );

}


/*
 * =========================================================
 * PARSE DECISION
 * =========================================================
 */


function parseSelection(
    raw,
    results
) {

    if (!raw) {

        return noSuitableSource(
            "Source Selector вернул пустой ответ"
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
            "Source Selector invalid JSON:",
            raw
        );


        return noSuitableSource(
            "Source Selector не смог надёжно оценить поисковую выдачу"
        );

    }


    const decision =
        String(
            parsed?.decision || ""
        )
            .trim()
            .toLowerCase();


    /*
     * =====================================================
     * REJECT
     * =====================================================
     */


    if (
        decision === "reject"
    ) {

        return noSuitableSource(
            parsed?.reason ||
            "Ни один найденный источник не удовлетворяет задаче"
        );

    }


    /*
     * =====================================================
     * SELECT
     * =====================================================
     */


    if (
        decision !== "select"
    ) {

        return noSuitableSource(
            "Source Selector вернул неизвестное решение"
        );

    }


    const index =
        Number(
            parsed?.index
        );


    if (
        !Number.isInteger(
            index
        ) ||
        index < 0 ||
        index >= results.length
    ) {

        return noSuitableSource(
            "Source Selector указал некорректный источник"
        );

    }


    return {
        success:
            true,

        noSuitableSource:
            false,

        index,

        result:
            results[index],

        reason:
            typeof parsed?.reason === "string"
                ? parsed.reason.trim()
                : ""
    };

}


/*
 * =========================================================
 * SELECT SOURCE
 * =========================================================
 */


export async function selectSource(
    task,
    searchResults
) {

    const results =
        normalizeResults(
            searchResults
        );


    /*
     * =====================================================
     * NO SEARCH RESULTS
     * =====================================================
     */


    if (
        results.length === 0
    ) {

        return noSuitableSource(
            "Поиск не вернул источники"
        );

    }


    /*
     * =====================================================
     * AI REQUIRED
     * =====================================================
     *
     * Больше НЕ используем автоматический result[0].
     *
     * Если Selector недоступен,
     * безопаснее отказаться от выбора,
     * чем загрузить заведомо неподходящий источник.
     */


    if (!groq) {

        return noSuitableSource(
            "Source Selector недоступен"
        );

    }


    /*
     * =====================================================
     * SOURCE EVALUATION
     * =====================================================
     */


    try {

        const response =
            await requestSelection(
                task,
                results
            );


        const raw =
            response
                ?.choices
                ?.[0]
                ?.message
                ?.content;


        const selection =
            parseSelection(
                raw,
                results
            );


        /*
         * =================================================
         * LOG
         * =================================================
         */


        if (
            selection.success
        ) {

            console.log(
                "Jessica Source Selector:",
                JSON.stringify({
                    decision:
                        "select",

                    index:
                        selection.index,

                    title:
                        selection.result?.title || "",

                    url:
                        selection.result?.url || "",

                    reason:
                        selection.reason || ""
                })
            );

        } else {

            console.warn(
                "Jessica Source Selector:",
                JSON.stringify({
                    decision:
                        "reject",

                    reason:
                        selection.reason || ""
                })
            );

        }


        return selection;


    } catch (error) {

        console.error(
            "Source Selector final error:",
            error
        );


        return noSuitableSource(
            error?.message ||
            "Не удалось выбрать надёжный источник"
        );

    }

            }
