import OpenAI from "openai";


/*
 * =========================================================
 * JESSICA SOURCE CONTENT VALIDATOR
 * =========================================================
 *
 * Проверяет, подходит ли реально загруженная страница
 * для ответа на исходную задачу.
 *
 * Не проверяет итоговый ответ.
 * Не выполняет поиск.
 * Не выполняет fetch.
 *
 * Только отвечает:
 *
 * "Хватает ли содержимого этой страницы?"
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
 * FIND FETCH RESULT
 * =========================================================
 */


function findFetchResult(
    taskRunResult
) {

    const results =
        Array.isArray(
            taskRunResult?.results
        )
            ? taskRunResult.results
            : [];


    for (
        let index =
            results.length - 1;

        index >= 0;

        index--
    ) {

        const result =
            results[index];


        if (
            result?.tool === "web_fetch" &&
            result?.success === true &&
            typeof result?.data?.content === "string" &&
            result.data.content.trim()
        ) {

            return result;
        }
    }


    return null;
}


/*
 * =========================================================
 * VALIDATE SOURCE CONTENT
 * =========================================================
 */


export async function validateSourceContent(
    task,
    plan,
    taskRunResult
) {

    /*
     * Проверка нужна только тогда,
     * когда Planner потребовал source_content.
     */
    if (
        plan?.evidence?.mode !==
        "source_content"
    ) {

        return {
            success: true,
            valid: true,
            shouldRetry: false,
            reason:
                "Проверка содержимого источника не требуется"
        };
    }


    const fetchResult =
        findFetchResult(
            taskRunResult
        );


    if (!fetchResult) {

        return {
            success: true,
            valid: false,
            shouldRetry: true,
            reason:
                "Не найдено успешно загруженное содержимое источника"
        };
    }


    const content =
        fetchResult.data.content.trim();


    const url =
        String(
            fetchResult.data?.url || ""
        ).trim();


    const title =
        String(
            fetchResult.data?.title || ""
        ).trim();


    /*
     * Без AI можем подтвердить только
     * сам факт наличия содержимого.
     *
     * Решение пока не блокируем.
     */
    if (!groq) {

        return {
            success: false,
            unavailable: true,
            valid: true,
            shouldRetry: false,
            reason:
                "AI-проверка содержимого источника недоступна"
        };
    }


    try {

        /*
         * Ограничиваем размер текста,
         * чтобы Validator не отправлял
         * огромную страницу целиком.
         */
        const contentForValidation =
            content.slice(
                0,
                18000
            );


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
                            "Ты Source Content Validator системы Jessica Core.",
                            "",
                            "Ты не отвечаешь пользователю.",
                            "Ты не придумываешь факты.",
                            "Ты проверяешь только переданное содержимое страницы.",
                            "",
                            "Определи, содержит ли эта страница достаточно информации,",
                            "чтобы выполнить исходную задачу.",
                            "",
                            "valid=true ставь только тогда, когда нужные данные",
                            "действительно присутствуют в содержимом страницы.",
                            "",
                            "Если страница относится к нужному сайту,",
                            "но нужного факта на ней нет, ставь valid=false.",
                            "",
                            "Если нужна другая или более конкретная страница,",
                            "ставь shouldRetry=true.",
                            "",
                            "Не считай страницу подходящей только потому,",
                            "что её домен выглядит официальным.",
                            "",
                            "Верни только JSON:",
                            JSON.stringify({
                                valid: true,
                                shouldRetry: false,
                                reason:
                                    "краткая причина"
                            })
                        ].join("\n")
                    },

                    {
                        role: "user",

                        content: [
                            "ИСХОДНАЯ ЗАДАЧА:",
                            String(task || ""),

                            "",
                            "URL:",
                            url,

                            "",
                            "TITLE:",
                            title,

                            "",
                            "СОДЕРЖИМОЕ СТРАНИЦЫ:",
                            contentForValidation
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

            return {
                success: false,
                unavailable: false,
                valid: true,
                shouldRetry: false,
                reason:
                    "Source Content Validator вернул пустой ответ"
            };
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
                "Source Content Validator invalid JSON:",
                raw
            );


            return {
                success: false,
                unavailable: false,
                valid: true,
                shouldRetry: false,
                reason:
                    "Source Content Validator вернул некорректный JSON"
            };
        }


        return {
            success: true,

            valid:
                parsed.valid === true,

            shouldRetry:
                parsed.shouldRetry === true,

            reason:
                typeof parsed.reason === "string"
                    ? parsed.reason
                    : ""
        };


    } catch (error) {

        console.error(
            "Source Content Validator error:",
            error
        );


        return {
            success: false,
            unavailable: true,
            valid: true,
            shouldRetry: false,
            reason:
                error?.message ||
                "Ошибка Source Content Validator"
        };
    }

}
