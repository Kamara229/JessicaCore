/*
 * =========================================================
 * JESSICA RELEASE FACT CANDIDATES
 * =========================================================
 */

import {
    RELEASE_SIGNAL
} from "./releaseFactConstants.js";


import {
    countSubjectMatches,
    getTitleIdentityText
} from "./releaseFactText.js";


import {
    extractDates,
    findNearestDate
} from "./releaseFactDates.js";


import {
    collectVersionMatches,
    compareVersions,
    isPrereleaseBound
} from "./releaseFactVersions.js";


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


export function buildCandidates({

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


export function selectCandidate(
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
