/*
 * =========================================================
 * JESSICA NATIVE DATE UTILS v1
 * =========================================================
 *
 * Детерминированный парсинг календарных дат.
 *
 * Используется native capabilities без AI:
 *
 * - date_diff tool;
 * - Native Skill Planner.
 *
 * =========================================================
 */


const MONTHS = new Map([
    ["января", 1],
    ["январь", 1],
    ["февраля", 2],
    ["февраль", 2],
    ["марта", 3],
    ["март", 3],
    ["апреля", 4],
    ["апрель", 4],
    ["мая", 5],
    ["май", 5],
    ["июня", 6],
    ["июнь", 6],
    ["июля", 7],
    ["июль", 7],
    ["августа", 8],
    ["август", 8],
    ["сентября", 9],
    ["сентябрь", 9],
    ["октября", 10],
    ["октябрь", 10],
    ["ноября", 11],
    ["ноябрь", 11],
    ["декабря", 12],
    ["декабрь", 12],

    ["january", 1],
    ["jan", 1],
    ["february", 2],
    ["feb", 2],
    ["march", 3],
    ["mar", 3],
    ["april", 4],
    ["apr", 4],
    ["may", 5],
    ["june", 6],
    ["jun", 6],
    ["july", 7],
    ["jul", 7],
    ["august", 8],
    ["aug", 8],
    ["september", 9],
    ["sep", 9],
    ["sept", 9],
    ["october", 10],
    ["oct", 10],
    ["november", 11],
    ["nov", 11],
    ["december", 12],
    ["dec", 12]
]);


function pad2(value)
{

    return String(value)
        .padStart(
            2,
            "0"
        );

}


function buildDate(
    year,
    month,
    day
) {

    const y =
        Number(year);

    const m =
        Number(month);

    const d =
        Number(day);


    if(
        !Number.isInteger(y)
        ||
        !Number.isInteger(m)
        ||
        !Number.isInteger(d)
        ||
        y < 1
        ||
        m < 1
        ||
        m > 12
        ||
        d < 1
        ||
        d > 31
    ){

        return null;

    }


    const date =
        new Date(
            Date.UTC(
                y,
                m - 1,
                d
            )
        );


    if(
        date.getUTCFullYear() !== y
        ||
        date.getUTCMonth() !== m - 1
        ||
        date.getUTCDate() !== d
    ){

        return null;

    }


    return {

        date,

        iso:
            `${y}-${pad2(m)}-${pad2(d)}`

    };

}


export function parseNativeDate(
    value
) {

    const text =
        String(
            value || ""
        )
            .trim()
            .toLowerCase()
            .replace(
                /\s+/g,
                " "
            );


    if(!text){

        return null;

    }


    /*
     * YYYY-MM-DD
     */


    let match =
        text.match(
            /(?:^|\D)(\d{4})-(\d{1,2})-(\d{1,2})(?:\D|$)/
        );


    if(match){

        return buildDate(
            match[1],
            match[2],
            match[3]
        );

    }


    /*
     * DD.MM.YYYY / DD-MM-YYYY / DD/MM/YYYY
     */


    match =
        text.match(
            /(?:^|\D)(\d{1,2})[./-](\d{1,2})[./-](\d{4})(?:\D|$)/
        );


    if(match){

        return buildDate(
            match[3],
            match[2],
            match[1]
        );

    }


    /*
     * 13 августа 2026
     * 13 August 2026
     */


    match =
        text.match(
            /(?:^|[^\p{L}\p{N}])(\d{1,2})\s+([\p{L}.]+)\s+(\d{4})(?:\s*г(?:ода|\.)?)?(?:[^\p{L}\p{N}]|$)/iu
        );


    if(match){

        const monthName =
            String(
                match[2] || ""
            )
                .replace(
                    /\./g,
                    ""
                )
                .toLowerCase();


        const month =
            MONTHS.get(
                monthName
            );


        if(month){

            return buildDate(
                match[3],
                month,
                match[1]
            );

        }

    }


    /*
     * August 13, 2026
     */


    match =
        text.match(
            /(?:^|[^\p{L}\p{N}])([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?[,]?\s+(\d{4})(?:[^\p{L}\p{N}]|$)/iu
        );


    if(match){

        const month =
            MONTHS.get(
                String(
                    match[1] || ""
                )
                    .toLowerCase()
            );


        if(month){

            return buildDate(
                match[3],
                month,
                match[2]
            );

        }

    }


    return null;

}


function pushDate(
    result,
    known,
    parsed
) {

    if(
        !parsed
        ||
        !parsed.iso
        ||
        known.has(
            parsed.iso
        )
    ){

        return;

    }


    known.add(
        parsed.iso
    );


    result.push(
        parsed
    );

}


export function extractNativeDates(
    value
) {

    const text =
        String(
            value || ""
        );


    if(!text){

        return [];

    }


    const result = [];

    const known =
        new Set();


    const patterns = [

        /\b\d{4}-\d{1,2}-\d{1,2}\b/g,

        /\b\d{1,2}[./-]\d{1,2}[./-]\d{4}\b/g,

        /\b\d{1,2}\s+[\p{L}.]+\s+\d{4}(?:\s*г(?:ода|\.)?)?/giu,

        /\b[a-z]+\s+\d{1,2}(?:st|nd|rd|th)?[,]?\s+\d{4}\b/giu

    ];


    for(
        const pattern
        of patterns
    ){

        const matches =
            text.match(
                pattern
            )
            ||
            [];


        for(
            const raw
            of matches
        ){

            pushDate(

                result,

                known,

                parseNativeDate(
                    raw
                )

            );

        }

    }


    return result;

}
