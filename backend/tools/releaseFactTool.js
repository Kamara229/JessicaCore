/*
 * =========================================================
 * JESSICA RELEASE FACT TOOL v2
 * =========================================================
 *
 * Детерминированно извлекает версию и дату релиза
 * из уже загруженного официального источника.
 *
 * Главные правила:
 *
 * - AI не используется;
 * - неизвестные значения не угадываются;
 * - prerelease (alpha/beta/rc/preview) не считается stable;
 * - предпочтение отдаётся секции Latest Releases / Releases;
 * - related software/news не должен подменять сам продукт;
 * - версия и дата должны быть подтверждены одним фрагментом.
 * =========================================================
 */


import {
    registerTool
} from "./toolRegistry.js";


import {
    parseNativeDate
} from "../core/native/dateUtils.js";


const MAX_CONTENT_LENGTH =
    50000;


const MAX_SECTION_LENGTH =
    10000;


const RELEASE_SIGNAL =
    /(latest|current|stable|release|released|version|releases|релиз|выпуск|верси|стабильн|актуальн|текущ)/iu;


const PRERELEASE_SIGNAL =
    /(?:alpha|beta|preview|pre-release|prerelease|release\s+candidate|\brc\b|devel|development)/iu;


const SEMVER_PATTERN =
    /\b(\d{1,3}(?:\.\d{1,4}){1,3})\b/g;


const SINGLE_MAJOR_PATTERN =
    /\b(?:version|jdk|java\s+se|postgresql|python|rust|node(?:\.js)?)\s+(\d{1,3})\b/giu;


const ISO_DATE_PATTERN =
    /\b(20\d{2})-(0[1-9]|1[0-2])-([0-2]\d|3[01])\b/g;


const SUBJECT_STOPWORDS =
    new Set([
        "найди","найти","на","официальном","официальный","официального",
        "сайте","сайт","дату","дата","выпуска","выпуск","релиза","релиз",
        "актуальной","актуальная","актуальный","стабильной","стабильная",
        "версии","версия","проверить","проверенную","укажи","определи",
        "последнюю","последняя","текущую","текущая","затем","рассчитай",
        "сколько","полных","дней","прошло","между","используй","найденную",
        "первой","части","the","a","an","find","official","website","site",
        "latest","current","stable","version","release","released","date",
        "source","verify","check","determine","most","recent","provide"
    ]);


/*
 * =========================================================
 * NORMALIZATION
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


function normalizeTitle(
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


function getSubjectTokens(
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


function getTitleIdentityText(
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


/*
 * =========================================================
 * SUBJECT MATCH
 * =========================================================
 */


function countSubjectMatches(
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


/*
 * =========================================================
 * RELEASE SECTION
 * =========================================================
 */


function isReleaseHeading(
    line
) {

    const value =

        String(
            line || ""
        )
            .replace(
                /^#{1,6}\s*/,
                ""
            )
            .trim();


    return /^(latest\s+releases?|current\s+releases?|release\s+history|releases?|последн\p{L}*\s+релиз\p{L}*|актуальн\p{L}*\s+релиз\p{L}*|релиз\p{L}*)$/iu
        .test(
            value
        );

}


function isMarkdownHeading(
    line
) {

    return /^#{1,6}\s+\S/u.test(
        String(
            line || ""
        )
    );

}


function extractPreferredReleaseSection(
    text
) {

    const lines =

        text.split(
            "\n"
        );


    let offset = 0;


    for(
        let index = 0;
        index < lines.length;
        index++
    ){

        const line =
            lines[index];


        if(
            !isReleaseHeading(
                line
            )
        ){

            offset +=
                line.length + 1;


            continue;

        }


        const start =
            offset;


        let end =
            Math.min(
                text.length,
                start + MAX_SECTION_LENGTH
            );


        let cursor =
            offset + line.length + 1;


        for(
            let next = index + 1;
            next < lines.length;
            next++
        ){

            const nextLine =
                lines[next];


            if(
                isMarkdownHeading(
                    nextLine
                )
            ){

                end =
                    Math.min(
                        end,
                        cursor
                    );


                break;

            }


            cursor +=
                nextLine.length + 1;

        }


        return {

            text:

                text.slice(
                    start,
                    end
                ),

            start,

            preferred:
                true

        };

    }


    return {

        text,

        start:
            0,

        preferred:
            false

    };

}


/*
 * =========================================================
 * DATES
 * =========================================================
 */


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


function extractDates(
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

        /\b[a-z]+\s+\d{1,2}(?:st|nd|rd|th)?[,]?\s+20\d{2}\b/giu;


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


function findNearestDate(
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


/*
 * =========================================================
 * VERSIONS
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


function isPrereleaseBound({

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

        /(?:alpha|beta|preview|pre-release|prerelease|release\s+candidate|rc|devel|development)\s*$/iu
            .test(
                before
            )

    );

}


/*
 * =========================================================
 * CANDIDATE LINES
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


function collectVersionMatches(
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


/*
 * =========================================================
 * CANDIDATE SCORING
 * =========================================================
 */


function buildCandidates({

    section,

    sourceTitle,

    sourceUrl,

    subjectTokens

}) {

    const text =
        section.text;


    const dates =

        extractDates(
            text
        );


    const ranges =

        buildLineRanges(
            text
        );


    const titleIdentity =

        getTitleIdentityText(

            sourceTitle,

            subjectTokens

        );


    const sourceSubjectMatches =

        countSubjectMatches(

            titleIdentity,

            subjectTokens

        );


    const candidates = [];


    for(
        const item
        of collectVersionMatches(
            text
        )
    ){

        if(
            isPrereleaseBound({

                text,

                version:
                    item.version,

                index:
                    item.index

            })
        ){

            continue;

        }


        const line =

            findLineForIndex(
                ranges,
                item.index
            );


        const lineText =

            line?.text || "";


        const nearby =

            text.slice(

                Math.max(
                    0,
                    item.index - 180
                ),

                Math.min(
                    text.length,
                    item.index + item.version.length + 260
                )

            );


        const lineSubjectMatches =

            countSubjectMatches(

                lineText,

                subjectTokens

            );


        const nearbySubjectMatches =

            countSubjectMatches(

                nearby,

                subjectTokens

            );


        const sameLineDate =

            dates.find(
                date =>
                    line
                    &&
                    date.index >= line.start
                    &&
                    date.index <= line.end
            )

            ||

            null;


        /*
         * Одночисловой major version разрешаем только
         * когда дата находится в той же строке.
         *
         * Это не позволяет фразе вроде
         * "PostgreSQL 19 is now available" забрать дату
         * соседнего stable-релиза 18.6.
         */


        if(
            item.kind === "major"
            &&
            !sameLineDate
        ){

            continue;

        }


        const nearestDate =

            sameLineDate

            ||

            findNearestDate(

                dates,

                item.index

            );


        if(!nearestDate){

            continue;

        }


        /*
         * Outside a dedicated release section, the candidate
         * must still be tied to the requested product by
         * line/near context or by official page identity.
         */


        if(
            section.preferred !== true
            &&
            subjectTokens.length > 0
            &&
            lineSubjectMatches === 0
            &&
            nearbySubjectMatches === 0
            &&
            sourceSubjectMatches === 0
        ){

            continue;

        }


        let score = 0;


        if(
            section.preferred === true
        ){

            score += 24;

        }


        if(
            RELEASE_SIGNAL.test(
                lineText
            )
        ){

            score += 7;

        }


        if(
            RELEASE_SIGNAL.test(
                nearby
            )
        ){

            score += 4;

        }


        score +=

            Math.min(
                12,
                lineSubjectMatches * 8
            );


        score +=

            Math.min(
                8,
                nearbySubjectMatches * 4
            );


        score +=

            Math.min(
                5,
                sourceSubjectMatches * 3
            );


        if(
            nearestDate.distance <= 80
        ){

            score += 12;

        }
        else if(
            nearestDate.distance <= 220
        ){

            score += 7;

        }
        else{

            score += 3;

        }


        if(
            /(?:[·|]\s*20\d{2}-\d{2}-\d{2}|20\d{2}-\d{2}-\d{2}\s*[·|])/u
                .test(
                    lineText
                )
        ){

            score += 8;

        }


        if(
            /\bnotes?\b/iu.test(
                lineText
            )
        ){

            score += 3;

        }


        candidates.push({

            version:
                item.version,

            kind:
                item.kind,

            index:
                item.index,

            score,

            releaseDate:
                nearestDate.value,

            rawDate:
                nearestDate.raw,

            dateIndex:
                nearestDate.index,

            line:
                lineText,

            subjectMatches:

                lineSubjectMatches
                +
                nearbySubjectMatches
                +
                sourceSubjectMatches

        });

    }


    return candidates;

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


    return [

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


                return (

                    left.index
                    -
                    right.index

                );

            }
        )[0]

        ||

        null;

}


/*
 * =========================================================
 * EVIDENCE
 * =========================================================
 */


function buildEvidenceFragment(
    sectionText,
    candidate
) {

    if(!candidate){

        return "";

    }


    const firstIndex =

        Math.min(
            candidate.index,
            candidate.dateIndex
        );


    const lastIndex =

        Math.max(
            candidate.index + candidate.version.length,
            candidate.dateIndex + String(candidate.rawDate || "").length
        );


    const start =

        Math.max(
            0,
            firstIndex - 220
        );


    const end =

        Math.min(
            sectionText.length,
            lastIndex + 420
        );


    return sectionText

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


function verifyEvidence(
    fragment,
    candidate
) {

    if(
        !fragment
        ||
        !fragment.includes(
            candidate.version
        )
    ){

        return false;

    }


    return extractDates(
        fragment
    )
        .some(
            date =>
                date.value ===
                candidate.releaseDate
        );

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


    const sourceTitle =

        normalizeTitle(
            args?.sourceTitle
        );


    const subject =

        normalizeText(
            args?.subject
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


    const subjectTokens =

        getSubjectTokens(
            subject
        );


    const preferredSection =

        extractPreferredReleaseSection(
            content
        );


    let candidates =

        buildCandidates({

            section:
                preferredSection,

            sourceTitle,

            sourceUrl,

            subjectTokens

        });


    /*
     * Если release section найден, но в нём нет
     * подтверждаемого stable fact, разрешаем fallback
     * на всю страницу, сохраняя subject identity checks.
     */


    if(
        candidates.length === 0
        &&
        preferredSection.preferred === true
    ){

        candidates =

            buildCandidates({

                section: {

                    text:
                        content,

                    start:
                        0,

                    preferred:
                        false

                },

                sourceTitle,

                sourceUrl,

                subjectTokens

            });

    }


    const selected =

        selectCandidate(
            candidates
        );


    if(
        !selected
        ||
        selected.score < 15
    ){

        return {

            success:
                false,

            text:
                "В источнике не удалось детерминированно подтвердить стабильную версию и дату выпуска именно запрошенного продукта.",

            data: {

                sourceUrl,

                sourceTitle,

                subjectTokens,

                releaseSectionFound:
                    preferredSection.preferred === true,

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

                                releaseDate:
                                    item.releaseDate,

                                score:
                                    item.score,

                                line:
                                    item.line

                            })
                        )

            }

        };

    }


    const evidenceFragment =

        buildEvidenceFragment(

            preferredSection.preferred === true

                ? preferredSection.text

                : content,

            selected

        );


    if(
        !verifyEvidence(
            evidenceFragment,
            selected
        )
    ){

        return {

            success:
                false,

            text:
                "Извлечённые версия и дата не подтверждаются одним фрагментом источника.",

            data: {

                sourceUrl,

                version:
                    selected.version,

                releaseDate:
                    selected.releaseDate

            }

        };

    }


    const sourceText =

        sourceUrl

            ? " Источник: " + sourceUrl

            : "";


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
                selected.releaseDate
                +
                "."
                +
                sourceText
            ),

        data: {

            subject,

            subjectTokens,

            version:
                selected.version,

            releaseDate:
                selected.releaseDate,

            sourceUrl,

            sourceTitle,

            evidenceVerified:
                true,

            evidenceFragment,

            releaseSectionFound:
                preferredSection.preferred === true,

            extractor:
                "release-fact-v2"

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
            "и дату выпуска запрошенного продукта из уже загруженного "
            +
            "официального источника. Не использует AI."
        ),

    arguments: {

        content:
            "Текст уже загруженного официального источника.",

        sourceUrl:
            "URL источника для provenance.",

        sourceTitle:
            "Заголовок загруженной страницы.",

        subject:
            "Исходная формулировка искомого продукта или релиза."

    },

    execute:
        executeReleaseFact

});
