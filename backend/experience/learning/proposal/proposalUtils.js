/*
 * =========================================================
 * JESSICA LEARNING PROPOSAL UTILS
 * =========================================================
 *
 * Общие функции нормализации.
 *
 * НЕ содержат бизнес-логику Learning.
 *
 * =========================================================
 */


export function isObject(
    value
) {

    return (

        value &&

        typeof value === "object" &&

        !Array.isArray(value)

    );

}


export function normalizeText(
    value
) {

    return String(
        value || ""
    )
    .trim();

}


export function normalizeNumber(
    value
) {

    const number =
        Number(value);


    return Number.isFinite(number)

        ? number

        : 0;

}


export function normalizeUnit(
    value
) {

    return Math.max(

        0,

        Math.min(

            1,

            normalizeNumber(
                value
            )

        )

    );

}


export function normalizePositiveInteger(
    value
) {

    const number =

        Math.floor(

            normalizeNumber(
                value
            )

        );


    return number > 0

        ? number

        : 0;

}


export function normalizeStringArray(
    value
) {

    if(
        !Array.isArray(
            value
        )
    ){

        return [];

    }


    const result = [];


    for(
        const item
        of value
    ){

        const normalized =

            normalizeText(
                item
            );


        if(
            !normalized
        ){

            continue;

        }


        if(
            result.includes(
                normalized
            )
        ){

            continue;

        }


        result.push(
            normalized
        );

    }


    return result;

}


export function normalizeObjectArray(
    value
) {

    if(
        !Array.isArray(
            value
        )
    ){

        return [];

    }


    return value

        .filter(
            isObject
        )

        .map(

            item => ({
                ...item
            })

        );

}
