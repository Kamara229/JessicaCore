/*
 * =========================================================
 * JESSICA CONFIDENCE NUMBER HELPERS
 * =========================================================
 */


export function normalizeNumber(
    value
) {

    const number =
        Number(value);


    return Number.isFinite(number)

        ? number

        : 0;

}


export function clampUnit(
    value
) {

    const number =

        normalizeNumber(
            value
        );


    return Math.max(

        0,

        Math.min(
            1,
            number
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
