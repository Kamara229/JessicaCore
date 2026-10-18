/*
 * =========================================================
 * JESSICA RELEASE FACT TOOL v1
 * =========================================================
 *
 * Детерминированно извлекает актуальную стабильную версию
 * и дату выпуска из уже загруженного официального источника.
 *
 * AI не используется.
 * =========================================================
 */


import {
    registerTool
} from "./toolRegistry.js";


const MAX_CONTENT_LENGTH =
    50000;


const RELEASE_SIGNAL =
    /(latest|current|stable|release|released|version|releases|релиз|выпуск|верси|стабильн|актуальн|текущ)/iu;


const PRERELEASE_SIGNAL =
    /(alpha|beta|preview|pre-release|prerelease|release\s+candidate|\brc\b|devel|development)/iu;


const ISO_DATE_PATTERN =
    /\b(20\d{2})-(0[1-9]|1[0-2])-([0-2]\d|3[01])\b/g;


const VERSION_PATTERN =
    /\b(\d{1,3}(?:\.\d{1,4}){1,3})\b/g;


/*
 * =========================================================
 * TEXT
 * =========================================================
 */


function normalizeText(
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


function normalizeUrl(
    value
) {

    return String(
        value || ""
    )
        .trim();

}


/*
 * =========================================================
 * VERSION
 * =========================================================
 */


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
        parts.length < 2
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


function compareVersions(
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


/*
 * =========================================================
 * DATE
 * =========================================================
 */


function isValidIsoDate(
    value
) {

    const match =

        String(
            value || ""
        )
            .match(
                /^(20\d{2})-(\d{2})-(\d{2})$/
            );


    if(!match){

        return false;

    }


    const year =
        Number(
            match[1]
        );


    const month =
        Number(
            match[2]
        );


    const day =
        Number(
            match[3]
        );


    const date =

        new Date(
            Date.UTC(
                year,
                month - 1,
                day
            )
        );


    return (

        date.getUTCFullYear() === year

        &&

        date.getUTCMonth() === month - 1

        &&

        date.getUTCDate() === day

    );

}


function extractDates(
    text
) {

    const result = [];

    const known =
        new Set();


    for(
        const match
        of text.matchAll(
            ISO_DATE_PATTERN
        )
    ){

        const value =
            match[0];


        if(
            !isValidIsoDate(
                value
            )
            ||
            known.has(
                value
            )
        ){

            continue;

        }


        known.add(
            value
        );


        result.push({

            value,

            index:
                match.index || 0

        });

    }


    return result;

}


/*
 * =========================================================
 * PRE-RELEASE
 * =========================================================
 */


function isPrereleaseVersion({

    text,

    version,

    index

}) {

    const before =

        text.slice(

            Math.max(
                0,
                index - 24
            ),

            index

        );


    const after =

        text.slice(

            index + version.length,

            Math.min(
                text.length,
                index + version.length + 24
            )

        );


    /*
     * Pre-release marker must belong to THIS version,
     * not merely appear elsewhere on the same line.
     *
     * Example:
     *
     * "19 Beta 3, 18.6, 17.11"
     *
     * Beta belongs to 19 and must not disqualify 18.6.
     */


    const afterPrerelease =

        /^\s*(?:alpha|beta|preview|pre-release|prerelease|rc|devel|development)\b/iu
            .test(
                after
            );


    const beforePrerelease =

        /(?:alpha|beta|preview|pre-release|prerelease|rc|devel|development)\s*$/iu
            .test(
                before
            );


    return (

        afterPrerelease

        ||

        beforePrerelease

    );

}


/*
 * =========================================================
 * LINE CONTEXT
 * =========================================================
 */


function buildLineRanges(
    text
) {

    const ranges = [];

    let cursor = 0;


    for(
        const line
        of text.split(
            "\n"
        )
    ){

        const start =
            cursor;


        const end =
            start + line.length;


        ranges.push({

            text:
                line.trim(),

            start,

            end

        });


        cursor =
            end + 1;

    }


    return ranges;

}


function findLineForIndex(
    ranges,
    index
) {

    return ranges.find(
        line =>
            index >= line.start
            &&
            index <= line.end
    )
    ||
    null;

}


function findNearestDate(
    dates,
    index,
    maxDistance = 900
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


/*
 * =========================================================
 * CANDIDATES
 * =========================================================
 */


function extractVersionCandidates(
    text,
    dates,
    ranges
) {

    const result = [];


    for(
        const match
        of text.matchAll(
            VERSION_PATTERN
        )
    ){

        const version =
            match[1];


        const index =
            match.index || 0;


        if(
            isPrereleaseVersion({

                text,

                version,

                index

            })
        ){

            continue;

        }


        const line =

            findLineForIndex(
                ranges,
                index
            );


        const lineText =

            line?.text || "";


        const nearbyText =

            text.slice(

                Math.max(
                    0,
                    index - 160
                ),

                Math.min(
                    text.length,
                    index + 260
                )

            );


        let score = 0;


        if(
            RELEASE_SIGNAL.test(
                lineText
            )
        ){

            score += 8;

        }


        if(
            RELEASE_SIGNAL.test(
                nearbyText
            )
        ){

            score += 4;

        }


        if(
            /latest releases?/iu.test(
                nearbyText
            )

            ||

            /последн\p{L}*\s+релиз/iu.test(
                nearbyText
            )
        ){

            score += 5;

        }


        if(
            /release date/iu.test(
                nearbyText
            )

            ||

            /дата\s+(?:релиза|выпуска)/iu.test(
                nearbyText
            )
        ){

            score += 5;

        }


        const nearestDate =

            findNearestDate(

                dates,

                index

            );


        if(nearestDate){

            score +=

                nearestDate.distance <= 160

                    ? 8

                    : 4;

        }


        result.push({

            version,

            index,

            score,

            nearestDate:

                nearestDate?.value

                ||

                null,

            distanceToDate:

                nearestDate?.distance

                ??

                null,

            line:
                lineText

        });

    }


    return result;

}


function selectCandidate(
    candidates
) {

    if(
        !Array.isArray(
            candidates
        )
        ||
        candidates.length === 0
    ){

        return null;

    }


    const ranked = [

        ...candidates

    ]
        .sort(
            (
                left,
                right
            ) => {

                if(
                    right.score !==
                    left.score
                ){

                    return (
                        right.score
                        -
                        left.score
                    );

                }


                const versionCompare =

                    compareVersions(

                        right.version,

                        left.version

                    );


                if(
                    versionCompare !== 0
                ){

                    return versionCompare;

                }


                const leftDistance =

                    left.distanceToDate

                    ??

                    Number.MAX_SAFE_INTEGER;


                const rightDistance =

                    right.distanceToDate

                    ??

                    Number.MAX_SAFE_INTEGER;


                return (
                    leftDistance
                    -
                    rightDistance
                );

            }
        );


    return ranked[0]

        ||

        null;

}


/*
 * =========================================================
 * EVIDENCE
 * =========================================================
 */


function buildEvidenceFragment(
    text,
    candidate
) {

    if(!candidate){

        return "";

    }


    const start =

        Math.max(
            0,
            candidate.index - 220
        );


    const end =

        Math.min(
            text.length,
            candidate.index + 520
        );


    return text

        .slice(
            start,
            end
        )
        .replace(
            /\s+/g,
            " "
        )
        .trim();

}


/*
 * =========================================================
 * EXECUTE
 * =========================================================
 */


function executeReleaseFact(
    args
) {

    const content =

        normalizeText(
            args?.content
        );


    const sourceUrl =

        normalizeUrl(
            args?.sourceUrl
        );


    if(!content){

        return {

            success:
                false,

            needsClarification:
                true,

            text:
                "Для извлечения версии и даты требуется содержимое источника.",

            data:
                null

        };

    }


    const dates =

        extractDates(
            content
        );


    const ranges =

        buildLineRanges(
            content
        );


    const candidates =

        extractVersionCandidates(

            content,

            dates,

            ranges

        );


    const selected =

        selectCandidate(
            candidates
        );


    if(
        !selected
        ||
        selected.score < 8
        ||
        !selected.nearestDate
    ){

        return {

            success:
                false,

            text:
                "В источнике не удалось детерминированно подтвердить стабильную версию и дату выпуска.",

            data: {

                sourceUrl,

                candidates:
                    candidates
                        .slice(
                            0,
                            8
                        )
                        .map(
                            item => ({

                                version:
                                    item.version,

                                score:
                                    item.score,

                                releaseDate:
                                    item.nearestDate

                            })
                        )

            }

        };

    }


    const evidenceFragment =

        buildEvidenceFragment(

            content,

            selected

        );


    const evidenceVerified =

        Boolean(
            evidenceFragment
        )

        &&

        evidenceFragment.includes(
            selected.version
        )

        &&

        evidenceFragment.includes(
            selected.nearestDate
        );


    if(!evidenceVerified){

        return {

            success:
                false,

            text:
                "Извлечённые данные не удалось подтвердить в одном фрагменте источника.",

            data: {

                sourceUrl,

                version:
                    selected.version,

                releaseDate:
                    selected.nearestDate

            }

        };

    }


    return {

        success:
            true,

        text:

            (
                "Актуальная стабильная версия — "
                +
                selected.version
                +
                ". Дата выпуска — "
                +
                selected.nearestDate
                +
                "."
            ),

        data: {

            version:
                selected.version,

            releaseDate:
                selected.nearestDate,

            sourceUrl,

            evidenceVerified:
                true,

            evidenceFragment,

            extractor:
                "release-fact-v1"

        }

    };

}


/*
 * =========================================================
 * REGISTER
 * =========================================================
 */


registerTool({

    name:
        "release_fact",

    description:

        (
            "Детерминированно извлекает актуальную стабильную версию "
            +
            "и дату выпуска из уже загруженного содержимого официального источника. "
            +
            "Не использует AI и не выполняет интернет-запрос."
        ),

    arguments: {

        content:
            "Текст уже загруженного официального источника.",

        sourceUrl:
            "URL источника для provenance."

    },

    execute:
        executeReleaseFact

});
