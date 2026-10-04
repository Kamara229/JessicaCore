/*
 * =========================================================
 * JESSICA PROPOSAL STORAGE UTILS
 * =========================================================
 *
 * Техническая нормализация.
 * Бизнес-логики здесь нет.
 *
 * =========================================================
 */


import {
    ALLOWED_PROPOSAL_STATUSES
} from "./proposalStorageConstants.js";


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


export function normalizeObject(
    value
) {

    return isObject(
        value
    )

        ? value

        : {};

}


export function normalizePositiveInteger(
    value,
    fallback = 1
) {

    const number =

        Math.floor(
            Number(value)
        );


    if(
        !Number.isInteger(number)
        ||
        number < 1
    ){

        return fallback;

    }


    return number;

}


export function normalizeProposalStatus(
    value
) {

    const status =

        normalizeText(
            value
        )
        .toUpperCase();


    return ALLOWED_PROPOSAL_STATUSES.includes(
        status
    )

        ? status

        : null;

}
