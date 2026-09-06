import OpenAI from "openai";


/*
 * =========================================================
 * JESSICA SOURCE SELECTOR
 * =========================================================
 *
 * Выбирает наиболее подходящий источник
 * из результатов web_search.
 *
 * Не выполняет поиск и не загружает страницы.
 * Только выбирает лучший результат.
 */


const groq =
    process.env.GROQ_API_KEY
        ? new OpenAI({
            apiKey: process.env.GROQ_API_KEY,
            baseURL: "https://api.groq.com/openai/v1"
        })
        : null;


/*
 * =========================================================
 * CLEAN JSON
 * =========================================================
 */


function cleanJsonText(text) {

    let value =
        String(text || "")
            .replace(/```json/gi, "")
            .replace(/```/g, "")
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


function normalizeResults(results) {

    if (!Array.isArray(results)) {
        return [];
    }


    return results
        .map(
            (item, index) => ({
                index,
                title:
                    String(item?.title || "").trim(),

                url:
                    String(item?.url || "").trim(),

                snippet:
                    String(item?.snippet || "").trim()
            })
        )
        .filter(
            item =>
                item.url
        );
}


/*
 * =========================================================
 * FALLBACK
 * =========================================================
 */


function fallbackSelection(results) {

    if (!results.length) {

        return {
            success: false,
            reason:
                "Нет результатов для выбора источника"
        };

    }


    return {
        success: true,

        index:
            0,

        result:
            results[0],

        reason:
            "Использован первый доступный результат"
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


    if (!results.length) {

        return {
            success: false,

            reason:
                "Поиск не вернул источники"
        };

    }


    /*
     * Если результат один,
     * выбирать не из чего.
     */
    if (
        results.length === 1
    ) {

        return {
            success: true,

            index:
                0,

            result:
                results[0],

            reason:
                "Доступен только один источник"
        };

    }


    /*
     * Если AI недоступен,
     * пока используем fallback.
     */
    if (!groq) {

        return fallbackSelection(
            results
        );

    }


    try {

        const response =
            await groq.chat.completions.create({

                model:
                    "openai/gpt-oss-20b",

                temperature:
                    0,

                messages: [
                    {
                        role: "system",

                        content: [
                            "Ты Source Selector системы Jessica Core.",
                            "Ты не отвечаешь пользователю.",
                            "Ты выбираешь один лучший источник из поисковых результатов.",
                            "",
                            "Оценивай источник по смыслу задачи.",
                            "",
                            "Если пользователь просит официальный источник,",
                            "предпочитай официальный сайт организации, компании, ведомства, проекта или автора.",
                            "",
                            "Если задача требует первичного источника,",
                            "предпочитай первоисточник, а не пересказ.",
                            "",
                            "Если нужен конкретный документ или страница,",
                            "выбирай результат, который наиболее вероятно содержит нужный материал.",
                            "",
                            "Не придумывай URL.",
                            "Можно выбирать только один из переданных результатов.",
                            "",
                            "Верни только JSON:",
                            JSON.stringify({
                                index: 0,
                                reason:
                                    "краткая причина выбора"
                            })
                        ].join("\n")
                    },

                    {
                        role: "user",

                        content: [
                            "ЗАДАЧА:",
                            String(task || ""),

                            "",
                            "РЕЗУЛЬТАТЫ ПОИСКА:",
                            JSON.stringify(
                                results,
                                null,
                                2
                            )
                        ].join("\n")
                    }
                ]

            });


        const raw =
            response
                ?.choices
                ?.[0]
                ?.message
                ?.content;


        if (!raw) {

            return fallbackSelection(
                results
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


            return fallbackSelection(
                results
            );

        }


        const index =
            Number(
                parsed.index
            );


        if (
            !Number.isInteger(index) ||
            index < 0 ||
            index >= results.length
        ) {

            return fallbackSelection(
                results
            );

        }


        return {
            success: true,

            index,

            result:
                results[index],

            reason:
                typeof parsed.reason === "string"
                    ? parsed.reason
                    : ""
        };


    } catch (error) {

        console.error(
            "Source Selector error:",
            error
        );


        return fallbackSelection(
            results
        );

    }

}
