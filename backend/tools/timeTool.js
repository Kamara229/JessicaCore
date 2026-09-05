import OpenAI from "openai";

import {
    registerTool
} from "./toolRegistry.js";


/*
 * =========================================================
 * CURRENT TIME TOOL
 * =========================================================
 *
 * Задача инструмента:
 *
 * 1. получить от Planner местоположение;
 * 2. определить IANA timezone;
 * 3. проверить timezone;
 * 4. взять реальное текущее время сервера;
 * 5. вернуть точный результат.
 *
 * Никаких списков городов и склонений.
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
 * TIMEZONE VALIDATION
 * =========================================================
 */


function isValidTimeZone(
    timeZone
) {

    if (
        typeof timeZone !== "string" ||
        !timeZone.trim()
    ) {

        return false;

    }


    try {

        new Intl.DateTimeFormat(
            "en-US",
            {
                timeZone
            }
        );


        return true;


    } catch {

        return false;

    }

}


/*
 * =========================================================
 * JSON CLEANUP
 * =========================================================
 */


function cleanJsonText(
    text
) {

    return text
        .replace(
            /```json/gi,
            ""
        )
        .replace(
            /```/g,
            ""
        )
        .trim();

}


/*
 * =========================================================
 * LOCATION → TIMEZONE
 * =========================================================
 *
 * AI здесь НЕ определяет текущее время.
 *
 * Он используется только как универсальный
 * языковой resolver:
 *
 * "в Дубае"
 * "Dubai"
 * "Ростове-на-Дону"
 * "New York"
 *
 * →
 *
 * Asia/Dubai
 * Europe/Moscow
 * America/New_York
 *
 * Само время затем вычисляет сервер.
 */


async function resolveTimeZone(
    location
) {

    /*
     * Если Planner уже передал IANA timezone,
     * дополнительный AI-вызов не нужен.
     */
    if (
        isValidTimeZone(
            location
        )
    ) {

        return {
            success: true,

            timeZone:
                location,

            location:
                location,

            confidence:
                1
        };

    }


    if (!groq) {

        return {
            success: false,

            text:
                "Невозможно определить часовой пояс: Groq не настроен"
        };

    }


    try {

        const response =
            await groq.responses.create({

                model:
                    "openai/gpt-oss-20b",

                instructions:
                    (
                        "Ты выполняешь только преобразование географического названия " +
                        "в IANA timezone. " +

                        "Не определяй текущее время. " +
                        "Не отвечай пользователю. " +

                        "Понимай названия городов и регионов на разных языках, " +
                        "включая падежи, сокращения и разговорные формы. " +

                        "Если место однозначно определяется, верни его IANA timezone. " +

                        "Если название действительно неоднозначно и без уточнения " +
                        "невозможно надёжно выбрать место, установи ambiguous=true. " +

                        "Не выдумывай timezone. " +

                        "Верни ТОЛЬКО JSON без markdown. " +

                        "Формат: " +

                        JSON.stringify({
                            location:
                                "нормализованное название места",
                            timeZone:
                                "IANA timezone или пустая строка",
                            ambiguous:
                                false,
                            confidence:
                                0.95
                        })
                    ),

                input:
                    `Местоположение: ${location}`,

                reasoning: {
                    effort:
                        "low"
                }

            });


        const raw =
            response.output_text
                ?.trim();


        if (!raw) {

            return {
                success: false,

                text:
                    "Resolver часового пояса вернул пустой ответ"
            };

        }


        let data;


        try {

            data =
                JSON.parse(
                    cleanJsonText(
                        raw
                    )
                );


        } catch {

            console.error(
                "Timezone resolver invalid JSON:",
                raw
            );


            return {
                success: false,

                text:
                    "Не удалось разобрать часовой пояс"
            };

        }


        if (
            data.ambiguous === true
        ) {

            return {
                success: false,

                needsClarification:
                    true,

                text:
                    `Местоположение «${location}» неоднозначно. Нужно уточнить город или регион.`
            };

        }


        const timeZone =
            typeof data.timeZone === "string"
                ? data.timeZone.trim()
                : "";


        if (
            !isValidTimeZone(
                timeZone
            )
        ) {

            console.error(
                "Invalid timezone returned:",
                timeZone
            );


            return {
                success: false,

                text:
                    `Не удалось надёжно определить часовой пояс для «${location}».`
            };

        }


        return {
            success: true,

            timeZone,

            location:
                typeof data.location === "string"
                    ? data.location.trim()
                    : location,

            confidence:
                typeof data.confidence === "number"
                    ? data.confidence
                    : null
        };


    } catch (error) {

        console.error(
            "Timezone resolver error:",
            error
        );


        return {
            success: false,

            text:
                "Ошибка определения часового пояса"
        };

    }

}


/*
 * =========================================================
 * GET CURRENT TIME
 * =========================================================
 */


async function executeCurrentTime(
    args
) {

    const location =
        typeof args?.location === "string"
            ? args.location.trim()
            : "";


    if (!location) {

        return {
            success: false,

            needsClarification:
                true,

            text:
                "Не указано местоположение для определения времени."
        };

    }


    const resolved =
        await resolveTimeZone(
            location
        );


    if (!resolved.success) {

        return resolved;

    }


    const now =
        new Date();


    const time =
        new Intl.DateTimeFormat(
            "ru-RU",
            {
                timeZone:
                    resolved.timeZone,

                hour:
                    "2-digit",

                minute:
                    "2-digit",

                second:
                    "2-digit",

                hour12:
                    false
            }
        ).format(
            now
        );


    const date =
        new Intl.DateTimeFormat(
            "ru-RU",
            {
                timeZone:
                    resolved.timeZone,

                weekday:
                    "long",

                day:
                    "2-digit",

                month:
                    "long",

                year:
                    "numeric"
            }
        ).format(
            now
        );


    return {
        success: true,

        data: {
            location:
                resolved.location,

            timeZone:
                resolved.timeZone,

            time,

            date,

            timestamp:
                now.toISOString()
        },

        text:
            (
                `Сейчас в ${resolved.location}: ${time}. ` +
                `${date}.`
            )
    };

}


/*
 * =========================================================
 * REGISTER
 * =========================================================
 */


registerTool({

    name:
        "current_time",

    description:
        (
            "Получает точное текущее время для любого города, региона " +
            "или IANA timezone. Используй, когда пользователь спрашивает, " +
            "сколько сейчас времени в каком-либо месте."
        ),

    arguments: {
        location:
            "Город, регион, страна или IANA timezone из запроса пользователя"
    },

    execute:
        executeCurrentTime

});
