/*
 * =========================================================
 * JESSICA RELEASE FACT DATES
 * =========================================================
 */

import {
    parseNativeDate
} from "../../core/native/dateUtils.js";


import {
    ISO_DATE_PATTERN
} from "./releaseFactConstants.js";


function pushParsedDate(
    result,
    seen,
    raw,
    index
) {

    const parsed =

        parseNativeDate(
            raw
        );


    if(
        !parsed?.iso
    ){

        return;

    }


    const key =

        parsed.iso
        +
        ":"
        +
        index;


    if(
        seen.has(
            key
        )
    ){

        return;

    }


    seen.add(
        key
    );


    result.push({

        value:
            parsed.iso,

        raw:

            String(
                raw || ""
            )
                .trim(),

        index

    });

}


export function extractDates(
    text
) {

    const result = [];

    const seen =
        new Set();


    for(
        const match
        of text.matchAll(
            ISO_DATE_PATTERN
        )
    ){

        pushParsedDate(
            result,
            seen,
            match[0],
            match.index || 0
        );

    }


    const dayFirstPattern =

        /\b\d{1,2}\s+[\p{L}.]+\s+20\d{2}(?:\s*г(?:ода|\.)?)?/giu;


    for(
        const match
        of text.matchAll(
            dayFirstPattern
        )
    ){

        pushParsedDate(
            result,
            seen,
            match[0],
            match.index || 0
        );

    }


    const monthFirstPattern =

        /\b[a-z]+\.?\s+\d{1,2}(?:st|nd|rd|th)?[,]?\s+20\d{2}\b/giu;


    for(
        const match
        of text.matchAll(
            monthFirstPattern
        )
    ){

        pushParsedDate(
            result,
            seen,
            match[0],
            match.index || 0
        );

    }


    return result;

}


export function findNearestDate(
    dates,
    index,
    maxDistance = 700
) {

    let best =
        null;


    for(
        const date
        of dates
    ){

        const distance =

            Math.abs(
                date.index - index
            );


        if(
            distance > maxDistance
        ){

            continue;

        }


        if(
            !best
            ||
            distance < best.distance
        ){

            best = {

                ...date,

                distance

            };

        }

    }


    return best;

}
