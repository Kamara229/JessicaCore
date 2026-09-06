import OpenAI from "openai";

import {
    executeAIWithRetry
} from "../../ai/aiRetry.js";


/*
 * =========================================================
 * JESSICA AI RESULT VALIDATOR
 * =========================================================
 *
 * Семантическая проверка качества ответа.
 *
 * Этот модуль:
 *
 * - не выполняет tools;
 * - не отвечает пользователю;
 * - не меняет план;
 * - только оценивает уже полученный результат.
 *
 * Технические ошибки AI:
 *
 * 429 / timeout / 5xx
 *
 * обрабатываются общим:
 *
 * ai/aiRetry.js
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


const VALIDATOR_MODEL =
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
 * AI REQUEST
 * =========================================================
 */


async function requestValidation(
    task,
    plan,
    taskRunResult,
    answerResult
) {

    return await executeAIWithRetry(
        async () => {

            return await groq
                .chat
                .completions
                .create({

                    model:
                        VALIDATOR_MODEL,

                    temperature:
                        0,

                    messages: [
                        {
                            role:
                                "system",

                            content: [
                                "Ты Validator системы Jessica Core.",

                                "",
                                "Ты не отвечаешь пользователю.",
                                "Ты не выполняешь инструменты.",
                                "Ты не меняешь план.",
                                "Ты только проверяешь уже сформированный ответ.",

                                "",
                                "Определи:",

                                "- решает ли ответ исходную задачу;",
                                "- соответствует ли ответ результатам выполнения;",
                                "- есть ли существенные неподтверждённые утверждения;",
                                "- может ли повторная попытка исправить проблему;",
                                "- требуется ли уточнение пользователя.",

                                "",
                                "Не требуй лишней детализации.",
                                "Не отклоняй хороший краткий ответ только потому, что его можно расширить.",

                                "",
                                "Если более ранние этапы уже получили данные из источников,",
                                "не придумывай новые факты и не выполняй собственный поиск.",

                                "",
                                "Верни только JSON:",

                                JSON.stringify({
                                    valid:
                                        true,

                                    shouldRetry:
                                        false,

                                    needsClarification:
                                        false,

                                    reason:
                                        "краткая причина"
                                })
                            ].join(
                                "\n"
                            )
                        },

                        {
                            role:
                                "user",

                            content: [
                                "ИСХОДНАЯ ЗАДАЧА:",
                                String(
                                    task || ""
                                ),

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
                            ].join(
                                "\n"
                            )
                        }
                    ]

                });

        },
        {
            label:
                "AI Validator"
        }
    );

}


/*
 * =========================================================
 * VALIDATE WITH AI
 * =========================================================
 */


export async function validateWithAI(
    task,
    plan,
    taskRunResult,
    answerResult
) {

    /*
     * =====================================================
     * AI UNAVAILABLE
     * =====================================================
     *
     * Главный validator.js сам решает,
     * что делать при недоступности
     * дополнительной AI-проверки.
     */


    if (!groq) {

        return {
            success:
                false,

            unavailable:
                true,

            reason:
                "AI Validator недоступен"
        };

    }


    /*
     * =====================================================
     * AI VALIDATION
     * =====================================================
     */


    try {

        const response =
            await requestValidation(
                task,
                plan,
                taskRunResult,
                answerResult
            );


        const raw =
            response
                ?.choices
                ?.[0]
                ?.message
                ?.content;


        /*
         * =================================================
         * EMPTY RESPONSE
         * =================================================
         */


        if (!raw) {

            return {
                success:
                    false,

                unavailable:
                    false,

                reason:
                    "AI Validator вернул пустой ответ"
            };

        }


        /*
         * =================================================
         * PARSE JSON
         * =================================================
         */


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
                success:
                    false,

                unavailable:
                    false,

                reason:
                    "AI Validator вернул некорректный JSON"
            };

        }


        /*
         * =================================================
         * SUCCESS
         * =================================================
         */


        return {
            success:
                true,

            valid:
                validation?.valid === true,

            shouldRetry:
                validation?.shouldRetry === true,

            needsClarification:
                validation?.needsClarification === true,

            reason:
                typeof validation?.reason === "string"
                    ? validation.reason.trim()
                    : ""
        };


    } catch (error) {

        /*
         * Сюда попадём только после того,
         * как aiRetry.js исчерпал технические
         * попытки либо получил ошибку,
         * которую повторять бессмысленно.
         */


        console.error(
            "AI Validator final error:",
            error
        );


        return {
            success:
                false,

            unavailable:
                true,

            status:
                error?.status || 0,

            reason:
                error?.message ||
                "Ошибка AI Validator"
        };

    }

}
