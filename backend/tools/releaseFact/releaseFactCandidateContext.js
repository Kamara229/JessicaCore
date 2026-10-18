/*
 * =========================================================
 * JESSICA RELEASE FACT CANDIDATE CONTEXT
 * =========================================================
 */

import {
    countSubjectMatches,
    getTitleIdentityText
} from "./releaseFactText.js";


import {
    extractDates,
    findNearestDate
} from "./releaseFactDates.js";


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


export function buildCandidateEnvironment({

    section,

    sourceTitle,

    subjectTokens

}) {

    return {

        text:
            section.text,

        dates:

            extractDates(
                section.text
            ),

        ranges:

            buildLineRanges(
                section.text
            ),

        sourceSubjectMatches:

            countSubjectMatches(

                getTitleIdentityText(
                    sourceTitle,
                    subjectTokens
                ),

                subjectTokens

            )

    };

}


export function buildCandidateContext({

    environment,

    item,

    subjectTokens

}) {

    const {
        text,
        dates,
        ranges,
        sourceSubjectMatches
    } = environment;


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

        return null;

    }


    const nearestDate =

        sameLineDate

        ||

        findNearestDate(
            dates,
            item.index
        );


    if(!nearestDate){

        return null;

    }


    return {

        lineText,

        nearby,

        lineSubjectMatches,

        nearbySubjectMatches,

        sourceSubjectMatches,

        nearestDate

    };

}
