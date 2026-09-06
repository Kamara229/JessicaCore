import OpenAI from "openai";


/*
 * =========================================================
 * JESSICA AI RESULT VALIDATOR
 * =========================================================
 *
 * Семантическая проверка качества ответа.
 *
 * Этот модуль:
 * - не выполняет tools;
 * - не отвечает пользователю;
 * - только оценивает уже полученный результат.
 */


const groq =
    process.env.GROQ_API_KEY
        ? new OpenAI({
            apiKey: process.env.GROQ_API_KEY,
            baseURL: "https://api.groq.com/openai/v1"
        })
        : null;


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


export async function validateWithAI(
    task,
    plan,
    taskRunResult,
    answerResult
) {

    /*
     * Если Groq недоступен,
     * этот модуль просто сообщает,
     * что AI-проверка не выполнена.
     *
     * Решение, считать ли это успехом,
     * будет принимать главный validator.js.
     */
    if (!groq) {

        return {
            success: false,
            unavailable: true,
            reason: "AI Validator недоступен"
        };

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
                            "Ты Validator системы Jessica Core.",
                            "Ты не отвечаешь пользователю.",
                            "Ты только проверяешь уже сформированный ответ.",
                            "",
                            "Определи:",
                            "- решает ли ответ исходную задачу;",
                            "- соответствует ли ответ результатам инструментов;",
                            "- есть ли существенные неподтверждённые утверждения;",
                            "- может ли повторная попытка исправить проблему;",
                            "- требуется ли уточнение пользователя.",
                            "",
                            "Не требуй лишней детализации.",
                            "Не отклоняй хороший краткий ответ только потому, что его можно расширить.",
                            "",
                            "Верни только JSON:",
                            JSON.stringify({
                                valid: true,
                                shouldRetry: false,
                                needsClarification: false,
                                reason: "краткая причина"
                            })
                        ].join("\n")
                    },

                    {
                        role: "user",

                        content: [
                            "ИСХОДНАЯ ЗАДАЧА:",
                            String(task || ""),

                            "",
                            "ПЛАН:",
                            JSON.stringify(
                                plan,
                                null,
                                2
                            ),

                            "",
                            "РЕЗУЛЬТАТ ВЫПОЛНЕНИЯ:",
                            JSON.stringify(
                                taskRunResult,
                                null,
                                2
                            ),

                            "",
                            "ИТОГОВЫЙ ОТВЕТ:",
                            String(
                                answerResult?.text || ""
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

            return {
                success: false,
                unavailable: false,
                reason:
                    "AI Validator вернул пустой ответ"
            };

        }


        let validation;


        try {

            validation =
                JSON.parse(
                    cleanJsonText(
                        raw
                    )
                );

        } catch {

            console.error(
                "AI Validator invalid JSON:",
                raw
            );


            return {
                success: false,
                unavailable: false,
                reason:
                    "AI Validator вернул некорректный JSON"
            };

        }


        return {
            success: true,

            valid:
                validation.valid === true,

            shouldRetry:
                validation.shouldRetry === true,

            needsClarification:
                validation.needsClarification === true,

            reason:
                typeof validation.reason === "string"
                    ? validation.reason
                    : ""
        };


    } catch (error) {

        console.error(
            "AI Validator error:",
            error
        );


        return {
            success: false,
            unavailable: true,
            reason:
                error?.message ||
                "Ошибка AI Validator"
        };

    }

}
