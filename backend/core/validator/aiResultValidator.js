/*
 * =========================================================
 * JESSICA AI RESULT VALIDATOR
 * =========================================================
 *
 * Проверяет качество уже сформированного ответа.
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
                    "- ищешь информацию;",
                    "- изменяешь план.",

                    "",

                    "Проверь:",

                    "- решает ли ответ исходную задачу;",
                    "- соответствует ли ответ данным выполнения;",
                    "- есть ли критические ошибки;",
                    "- нужна ли повторная попытка;",
                    "- требуется ли уточнение пользователя.",

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



    } catch(error) {


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
