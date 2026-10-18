/*
 * =========================================================
 * JESSICA RELEASE FACT VERSIONS
 * =========================================================
 */

import {
    PRERELEASE_SIGNAL,
    SEMVER_PATTERN,
    SINGLE_MAJOR_PATTERN
} from "./releaseFactConstants.js";


function parseVersion(
    value
) {

    const parts =

        String(
            value || ""
        )
            .split(
                "."
            )
            .map(
                item =>
                    Number(item)
            );


    if(
        parts.length === 0
        ||
        parts.some(
            item =>
                !Number.isInteger(item)
                ||
                item < 0
        )
    ){

        return null;

    }


    return parts;

}


export function compareVersions(
    left,
    right
) {

    const a =
        parseVersion(
            left
        );


    const b =
        parseVersion(
            right
        );


    if(
        !a
        ||
        !b
    ){

        return 0;

    }


    const length =

        Math.max(
            a.length,
            b.length
        );


    for(
        let index = 0;
        index < length;
        index++
    ){

        const av =
            a[index] || 0;


        const bv =
            b[index] || 0;


        if(
            av !== bv
        ){

            return av - bv;

        }

    }


    return 0;

}


export function isPrereleaseBound({

    text,

    version,

    index

}) {

    const before =

        text.slice(

            Math.max(
                0,
                index - 28
            ),

            index

        );


    const after =

        text.slice(

            index + version.length,

            Math.min(
                text.length,
                index + version.length + 34
            )

        );


    return (

        /^\s*(?:alpha|beta|preview|pre-release|prerelease|release\s+candidate|rc|devel|development)\b/iu
            .test(
                after
            )

        ||

        new RegExp(
            PRERELEASE_SIGNAL.source + "\\s*$",
            "iu"
        )
            .test(
                before
            )

    );

}


export function collectVersionMatches(
    text
) {

    const result = [];

    const seen =
        new Set();


    const push = (
        version,
        index,
        kind
    ) => {

        const key =

            version
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

            version,

            index,

            kind

        });

    };


    for(
        const match
        of text.matchAll(
            SEMVER_PATTERN
        )
    ){

        push(
            match[1],
            match.index || 0,
            "semver"
        );

    }


    for(
        const match
        of text.matchAll(
            SINGLE_MAJOR_PATTERN
        )
    ){

        const full =
            match[0];


        const version =
            match[1];


        const localIndex =

            full.lastIndexOf(
                version
            );


        push(

            version,

            (match.index || 0)
            +
            Math.max(
                0,
                localIndex
            ),

            "major"

        );

    }


    return result;

}
