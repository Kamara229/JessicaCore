import {
    registerTool
} from "./toolRegistry.js";


import {
    isTimezoneResolverAvailable,
    requestTimezoneResolution
} from "../ai/timezoneResolverClient.js";


/*
 * =========================================================
 * CURRENT TIME TOOL v2
 * =========================================================
 *
 * Flow:
 *
 * location
 *      ↓
 * already IANA?
 *      │
 *      ├── yes
 *      │
 *      └── no
 *           ↓
 * Timezone Resolver AI
 *           ↓
 * validated IANA timezone
 *           ↓
 * server Date
 *           ↓
 * Intl.DateTimeFormat
 *           ↓
 * exact current time
 *
 *
 * КРИТИЧЕСКО:
 *
 * AI никогда не определяет текущее время.
 *
 * AI используется только для преобразования
 * свободного географического названия
 * в IANA timezone.
 *
 * =========================================================
 */


/*
 * =========================================================
 * TIMEZONE VALIDATION
 * =========================================================
 */


function isValidTimeZone(
    timeZone
) {

    if(
        typeof timeZone !== "string"
        ||
        !timeZone.trim()
    ){

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


    }catch(error){


        return false;

    }

}


/*
 * =========================================================
 * CLEAN JSON
 * =========================================================
 */


function cleanJsonText(
    text
) {

    if(
        typeof text !== "string"
    ){

        return "";

    }


    let value =

        text
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


    if(
        firstBrace !== -1
        &&
        lastBrace > firstBrace
    ){

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
 * RESOLVER INSTRUCTIONS
 * =========================================================
 */


function buildTimezoneResolverInstructions()
{

    return (

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

    );

}


/*
 * =========================================================
 * LOCATION → TIMEZONE
 * =========================================================
 */


async function resolveTimeZone(
    location
) {

    /*
     * =====================================================
     * ALREADY IANA
     * =====================================================
     */


    if(
        isValidTimeZone(
            location
        )
    ){

        return {

            success:
                true,

            timeZone:
                location,

            location,

            confidence:
                1

        };

    }


    /*
     * =====================================================
     * AI AVAILABILITY
     * =====================================================
     */


    if(
        !isTimezoneResolverAvailable()
    ){

        return {

            success:
                false,

            text:
                "Невозможно определить часовой пояс: Groq не настроен"

        };

    }


    /*
     * =====================================================
     * AI RESOLUTION
     * =====================================================
     */


    try {


        const response =

            await requestTimezoneResolution({

                instructions:

                    buildTimezoneResolverInstructions(),

                location

            });


        const raw =

            response
                ?.output_text
                ?.trim();


        if(
            !raw
        ){

            return {

                success:
                    false,

                text:
                    "Resolver часового пояса вернул пустой ответ"

            };

        }


        /*
         * =================================================
         * PARSE
         * =================================================
         */


        let data;


        try {


            data =

                JSON.parse(

                    cleanJsonText(
                        raw
                    )

                );


        }catch(error){


            console.error(

                "Timezone resolver invalid JSON:",

                raw

            );


            return {

                success:
                    false,

                text:
                    "Не удалось разобрать часовой пояс"

            };

        }


        /*
         * =================================================
         * AMBIGUOUS
         * =================================================
         */


        if(
            data?.ambiguous === true
        ){

            return {

                success:
                    false,

                needsClarification:
                    true,

                text:

                    `Местоположение «${location}» неоднозначно. ` +
                    "Нужно уточнить город или регион."

            };

        }


        /*
         * =================================================
         * TIMEZONE
         * =================================================
         */


        const timeZone =

            typeof data?.timeZone === "string"

                ? data.timeZone.trim()

                : "";


        if(
            !isValidTimeZone(
                timeZone
            )
        ){

            console.error(

                "Invalid timezone returned:",

                timeZone

            );


            return {

                success:
                    false,

                text:

                    `Не удалось надёжно определить часовой пояс для «${location}».`

            };

        }


        return {

            success:
                true,

            timeZone,

            location:

                typeof data?.location === "string"
                &&
                data.location.trim()

                    ? data.location.trim()

                    : location,

            confidence:

                typeof data?.confidence === "number"

                    ? data.confidence

                    : null

        };


    }catch(error){


        console.error(

            "Timezone resolver final error:",

            {

                status:
                    error?.status || 0,

                code:
                    error?.code || null,

                message:

                    error?.message

                    ||

                    String(error)

            }

        );


        return {

            success:
                false,

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


    if(
        !location
    ){

        return {

            success:
                false,

            needsClarification:
                true,

            text:
                "Не указано местоположение для определения времени."

        };

    }


    /*
     * =====================================================
     * TIMEZONE
     * =====================================================
     */


    const resolved =

        await resolveTimeZone(
            location
        );


    if(
        !resolved.success
    ){

        return resolved;

    }


    /*
     * =====================================================
     * SERVER CLOCK
     * =====================================================
     */


    const now =
        new Date();


    /*
     * =====================================================
     * TIME
     * =====================================================
     */


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

        )
            .format(
                now
            );


    /*
     * =====================================================
     * DATE
     * =====================================================
     */


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

        )
            .format(
                now
            );


    /*
     * =====================================================
     * RESULT
     * =====================================================
     */


    return {

        success:
            true,

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
                `Сейчас в ${resolved.location}: ${time}. `
                +
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
