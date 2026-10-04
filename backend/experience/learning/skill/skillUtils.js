/*
 * =========================================================
 * JESSICA EXPERIENCE SKILL UTILS
 * =========================================================
 *
 * Общие функции нормализации
 * Experience Skill.
 *
 * Бизнес-логики Learning здесь нет.
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


export function normalizeVersion(
    value
) {

    const version =
        Number(value);


    if(
        !Number.isInteger(version)
        ||
        version < 1
    ){

        return 1;

    }


    return version;

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
            !result.includes(
                normalized
            )
        ){

            result.push(
                normalized
            );

        }

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
