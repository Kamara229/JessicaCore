/*
 * =========================================================
 * JESSICA DATE DIFF TOOL v1
 * =========================================================
 *
 * Детерминированный расчёт количества
 * полных календарных дней между двумя датами.
 *
 * AI не используется.
 *
 * =========================================================
 */


import {
    registerTool
} from "./toolRegistry.js";


import {
    parseNativeDate
} from "../core/native/dateUtils.js";


const DAY_MS =
    24 * 60 * 60 * 1000;


function executeDateDiff(
    args
) {

    const start =

        parseNativeDate(
            args?.startDate
        );


    const end =

        parseNativeDate(
            args?.endDate
        );


    if(
        !start
        ||
        !end
    ){

        return {

            success:
                false,

            needsClarification:
                true,

            text:
                "Для расчёта нужны две корректные календарные даты."

        };

    }


    const milliseconds =

        end.date.getTime()

        -

        start.date.getTime();


    const signedDays =

        Math.trunc(
            milliseconds / DAY_MS
        );


    const absoluteDays =

        Math.abs(
            signedDays
        );


    return {

        success:
            true,

        data: {

            startDate:
                start.iso,

            endDate:
                end.iso,

            signedDays,

            fullDays:
                absoluteDays

        },

        text:

            (
                "Между "
                +
                start.iso
                +
                " и "
                +
                end.iso
                +
                " прошло "
                +
                absoluteDays
                +
                " полных дней."
            )

    };

}


registerTool({

    name:
        "date_diff",

    description:

        (
            "Детерминированно рассчитывает количество полных дней "
            +
            "между двумя календарными датами без AI. "
            +
            "Используй для разницы между датами, когда обе даты уже известны."
        ),

    arguments: {

        startDate:
            "Первая календарная дата.",

        endDate:
            "Вторая календарная дата."

    },

    execute:
        executeDateDiff

});
