/*
 * =========================================================
 * JESSICA SOURCE CONTENT VALIDATOR
 * =========================================================
 *
 * Проверяет качество реально загруженного источника.
 *
 *
 * Проверяет:
 *
 * task
 *   +
 * web_fetch content
 *        ↓
 * подходит ли источник
 *
 *
 * НЕ:
 *
 * - ищет источник;
 * - выполняет fetch;
 * - проверяет итоговый ответ;
 * - проверяет claims.
 *
 *
 * AI:
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


    const start =
        value.indexOf("{");


    const end =
        value.lastIndexOf("}");


    if (
        start !== -1 &&
        end > start
    ) {

        value =
            value.slice(
                start,
                end + 1
            );

    }


    return value;

}



/*
 * =========================================================
 * FIND FETCH CONTENT
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



    return (

        results
            .slice()
            .reverse()
            .find(

                item =>

                    item?.tool === "web_fetch" &&

                    item?.success !== false &&

                    typeof item?.data?.content === "string" &&

                    item.data.content.trim()

            )

        || null

    );

}



/*
 * =========================================================
 * BUILD PROMPT
 * =========================================================
 */


function buildMessages(
    task,
    url,
    title,
    content
) {


    return [

        {

            role:
                "system",

            content:
                [

                    "Ты Source Content Validator системы Jessica Core.",

                    "",

                    "Ты проверяешь только переданное содержимое страницы.",

                    "",

                    "Не выполняй поиск.",

                    "Не придумывай факты.",

                    "Не отвечай пользователю.",

                    "",

                    "Определи:",

                    "- содержит ли страница нужную информацию;",
                    "- достаточно ли её для выполнения задачи.",

                    "",

                    "Если страница официальная, но нужных данных нет — valid=false.",

                    "",

                    "Верни только JSON:",

                    JSON.stringify({

                        valid:
                            true,

                        shouldRetry:
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

                    "ЗАДАЧА:",

                    String(
                        task || ""
                    ),


                    "",


                    "URL:",

                    url,


                    "",


                    "TITLE:",

                    title,


                    "",


                    "СОДЕРЖИМОЕ:",

                    content.slice(
                        0,
                        18000
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
 * AI CHECK
 * =========================================================
 */


async function requestSourceValidation(
    task,
    url,
    title,
    content
) {


    return executeAIWithRetry(

        async () => {


            return validatorChat(

                buildMessages(
                    task,
                    url,
                    title,
                    content
                )

            );


        },

        {
            label:
                "Source Content Validator"
        }

    );

}



/*
 * =========================================================
 * PUBLIC
 * =========================================================
 */


export async function validateSourceContent(

    task,

    plan,

    taskRunResult

) {


    if (
        plan?.evidence?.mode !==
        "source_content"
    ) {

        return {

            success:
                true,

            valid:
                true,

            shouldRetry:
                false,

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

            success:
                true,

            valid:
                false,

            shouldRetry:
                true,

            reason:
                "Источник не был загружен"

        };

    }



    const content =
        fetchResult.data.content.trim();


    const url =
        String(
            fetchResult.data.url || ""
        );


    const title =
        String(
            fetchResult.data.title || ""
        );



    try {


        const response =
            await requestSourceValidation(

                task,

                url,

                title,

                content

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

                valid:
                    true,

                shouldRetry:
                    false,

                reason:
                    "Пустой ответ Source Validator"

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


            return {

                success:
                    false,

                valid:
                    true,

                shouldRetry:
                    false,

                reason:
                    "Некорректный JSON Source Validator"

            };

        }



        return {

            success:
                true,

            valid:
                parsed.valid === true,

            shouldRetry:
                parsed.shouldRetry === true,

            reason:
                typeof parsed.reason === "string"
                    ? parsed.reason.trim()
                    : ""

        };


    } catch(error) {


        console.error(
            "Source Content Validator error:",
            error
        );


        return {

            success:
                false,

            unavailable:
                true,

            valid:
                true,

            shouldRetry:
                false,

            reason:
                error?.message ||
                "Ошибка Source Content Validator"

        };

    }

}
