/*
 * =========================================================
 * JESSICA DATE DIFF TOOL v2
 * =========================================================
 *
 * Детерминированный расчёт количества
 * полных календарных дней между двумя датами.
 *
 * Режимы:
 *
 * difference — абсолютная разница между датами;
 * elapsed    — сколько дней прошло от startDate к endDate.
 *
 * В elapsed конечная дата не может быть раньше начальной.
 * AI не используется.
 * =========================================================
 */


import {
    registerTool
} from "./toolRegistry.js";


import {
    parseNativeDate
} from "../core/native/dateUtils.js";

import {formatDateDifference} from "./dateDiffResultText.js";


const DAY_MS =
    24 * 60 * 60 * 1000;


function normalizeMode(
    value
) {

    const mode =

        String(
            value || ""
        )
            .trim()
            .toLowerCase();


    return mode === "elapsed"

        ? "elapsed"

        : "difference";

}


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


    const mode =

        normalizeMode(
            args?.mode
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
                "Для расчёта нужны две корректные календарные даты.",

            data:
                null

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


    if(
        mode === "elapsed"
        &&
        signedDays < 0
    ){

        return {

            success:
                false,

            needsClarification:
                false,

            text:

                (
                    "Конечная дата "
                    +
                    end.iso
                    +
                    " раньше начальной даты "
                    +
                    start.iso
                    +
                    ". Нельзя утверждать, что между ними уже прошло положительное число дней в заданном направлении."
                ),

            data: {

                startDate:
                    start.iso,

                endDate:
                    end.iso,

                signedDays,

                mode,

                chronologyValid:
                    false

            }

        };

    }


    const fullDays =

        mode === "elapsed"

            ? signedDays

            : Math.abs(
                signedDays
            );


    const text = formatDateDifference(start.iso, end.iso, fullDays, mode);

    return {

        success:
            true,

        data: {

            startDate:
                start.iso,

            endDate:
                end.iso,

            signedDays,

            fullDays,

            mode,

            chronologyValid:
                true

        },

        text

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
            "Используй mode=elapsed, когда пользователь спрашивает, "
            +
            "сколько дней прошло от первой даты ко второй."
        ),

    arguments: {

        startDate:
            "Первая календарная дата.",

        endDate:
            "Вторая календарная дата.",

        mode:
            "difference для абсолютной разницы или elapsed для направленного интервала."

    },

    execute:
        executeDateDiff

});
