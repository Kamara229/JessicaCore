/*
 * =========================================================
 * JESSICA AI RESULT VALIDATOR
 * =========================================================
 *
 * Проверяет качество уже сформированного ответа.
 *
 *
 * Дополнительно определяет semantic outcome:
 *
 * result
 * → искомый результат найден / подтверждён.
 *
 * no_verified_result
 * → ответ корректен, но достоверный результат
 *   подтвердить не удалось.
 *
 *
 * Важно:
 *
 * valid и outcomeType — разные понятия.
 *
 * Например:
 *
 * {
 *   valid: true,
 *   outcomeType: "no_verified_result"
 * }
 *
 * означает:
 *
 * Jessica корректно сообщила,
 * что подтверждённых данных не найдено.
 *
 *
 * НЕ:
 *
 * - выполняет инструменты;
 * - строит план;
 * - меняет ответ;
 * - вызывает Groq напрямую.
 *
 *
 * AI доступ:
 *
 * validatorClient.js
 *
 * =========================================================
 */


import {
    validatorChat
} from "../../ai/validatorClient.js";

import {
    executeAIWithRetry
} from "../../ai/aiRetry.js";


/*
 * =========================================================
 * OUTCOME TYPES
 * =========================================================
 */


const OUTCOME_RESULT =
    "result";


const OUTCOME_NO_VERIFIED_RESULT =
    "no_verified_result";


/*
 * =========================================================
 * CLEAN JSON
 * =========================================================
 */


function cleanJsonText(
    text
) {

    let value =
        String(
            text || ""
        )
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
 * NORMALIZE OUTCOME TYPE
 * =========================================================
 */


function normalizeOutcomeType(
    value
) {

    if (
        value ===
        OUTCOME_NO_VERIFIED_RESULT
    ) {

        return OUTCOME_NO_VERIFIED_RESULT;

    }


    return OUTCOME_RESULT;

}


/*
 * =========================================================
 * PROMPT
 * =========================================================
 */


function buildValidatorMessages(
    task,
    plan,
    taskRunResult,
    answerResult
) {

    return [

        {

            role:
                "system",

            content:
                [

                    "Ты AI Validator системы Jessica Core.",

                    "",

                    "Ты проверяешь только уже полученный результат.",

                    "",

                    "Ты НЕ:",

                    "- отвечаешь пользователю;",
                    "- выполняешь инструменты;",
                    "- ищешь новую информацию;",
                    "- изменяешь план.",

                    "",

                    "Проверь:",

                    "- решает ли ответ исходную задачу;",
                    "- соответствует ли ответ фактическим данным выполнения;",
                    "- есть ли критические ошибки;",
                    "- нужна ли повторная попытка;",
                    "- требуется ли уточнение пользователя.",

                    "",

                    "Также определи outcomeType.",

                    "",

                    "Допустимы только два outcomeType:",

                    "",

                    "1. result",

                    "Используй result, если в данных выполнения есть достаточные основания считать, что искомый пользователем результат найден, установлен или подтверждён.",

                    "",

                    "2. no_verified_result",

                    "Используй no_verified_result, если итоговый ответ корректно сообщает, что искомый результат не удалось найти, подтвердить или установить по имеющимся данным.",

                    "",

                    "Примеры no_verified_result:",

                    "- официальный сайт не удалось подтвердить;",
                    "- поиск не дал сведений об указанном проекте;",
                    "- среди найденных источников нет достоверного подтверждения;",
                    "- данных недостаточно для подтверждения конкретного результата, и ответ честно это сообщает.",

                    "",

                    "Важно:",

                    "- no_verified_result НЕ означает, что ответ неправильный;",
                    "- valid может быть true одновременно с outcomeType=no_verified_result;",
                    "- отсутствие подтверждения нельзя превращать в выдуманный положительный результат;",
                    "- не считай сам факт наличия поисковой выдачи подтверждённым результатом;",
                    "- если найден только сторонний или неподтверждённый источник, это не result;",
                    "- если ответ содержит подтверждённый искомый результат, используй result.",

                    "",

                    "Не отклоняй короткий корректный ответ.",

                    "",

                    "Верни только JSON:",

                    JSON.stringify({

                        valid:
                            true,

                        shouldRetry:
                            false,

                        needsClarification:
                            false,

                        outcomeType:
                            "result",

                        reason:
                            "краткая причина"

                    })

                ]
                    .join(
                        "\n"
                    )

        },


        {

            role:
                "user",

            content:
                [

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

                ]
                    .join(
                        "\n"
                    )

        }

    ];

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

            return await validatorChat(

                buildValidatorMessages(

                    task,

                    plan,

                    taskRunResult,

                    answerResult

                )

            );

        },

        {

            label:
                "AI Validator"

        }

    );

}


/*
 * =========================================================
 * PUBLIC
 * =========================================================
 */


export async function validateWithAI(

    task,

    plan,

    taskRunResult,

    answerResult

) {

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
         * PARSE
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
         * NORMALIZED RESULT
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


            outcomeType:
                normalizeOutcomeType(
                    validation?.outcomeType
                ),


            reason:
                typeof validation?.reason === "string"
                    ? validation.reason.trim()
                    : ""

        };

    } catch (error) {

        console.error(
            "AI Validator error:",
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
