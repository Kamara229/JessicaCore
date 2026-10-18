/*
 * =========================================================
 * JESSICA RELEASE FACT TEXT
 * =========================================================
 */

import {
    MAX_CONTENT_LENGTH,
    SUBJECT_STOPWORDS
} from "./releaseFactConstants.js";


export function normalizeText(
    value
) {

    return String(
        value || ""
    )
        .replace(
            /\u00a0/g,
            " "
        )
        .replace(
            /\r/g,
            ""
        )
        .trim()
        .slice(
            0,
            MAX_CONTENT_LENGTH
        );

}


export function normalizeUrl(
    value
) {

    return String(
        value || ""
    )
        .trim();

}


export function normalizeTitle(
    value
) {

    return String(
        value || ""
    )
        .replace(
            /\s+/g,
            " "
        )
        .trim();

}


function tokenize(
    value
) {

    return String(
        value || ""
    )
        .toLowerCase()
        .match(
            /[\p{L}\p{N}][\p{L}\p{N}._-]*/gu
        )
        || [];

}


export function getSubjectTokens(
    subject
) {

    const seen =
        new Set();


    const result = [];


    for(
        const token
        of tokenize(
            subject
        )
    ){

        const clean =

            token
                .replace(
                    /^[._-]+|[._-]+$/g,
                    ""
                );


        if(
            clean.length < 3
            ||
            SUBJECT_STOPWORDS.has(
                clean
            )
            ||
            /^\d+$/.test(
                clean
            )
            ||
            seen.has(
                clean
            )
        ){

            continue;

        }


        seen.add(
            clean
        );


        result.push(
            clean
        );

    }


    return result.slice(
        0,
        8
    );

}


export function getTitleIdentityText(
    title,
    subjectTokens
) {

    const normalized =

        normalizeTitle(
            title
        );


    if(!normalized){

        return "";

    }


    const colonIndex =

        normalized.indexOf(
            ":"
        );


    if(
        colonIndex > 0
    ){

        const prefix =

            normalized
                .slice(
                    0,
                    colonIndex
                )
                .toLowerCase();


        const brandingOnly =

            Array.isArray(
                subjectTokens
            )
            &&
            subjectTokens.some(
                token =>
                    prefix.includes(
                        token
                    )
            );


        if(brandingOnly){

            return normalized

                .slice(
                    colonIndex + 1
                )
                .trim();

        }

    }


    return normalized;

}


export function countSubjectMatches(
    text,
    tokens
) {

    if(
        !text
        ||
        !Array.isArray(tokens)
        ||
        tokens.length === 0
    ){

        return 0;

    }


    const normalized =

        String(
            text
        )
            .toLowerCase();


    let count = 0;


    for(
        const token
        of tokens
    ){

        if(
            normalized.includes(
                token
            )
        ){

            count++;

        }

    }


    return count;

}
