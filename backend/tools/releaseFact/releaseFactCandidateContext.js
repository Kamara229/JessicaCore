/*
 * =========================================================
 * JESSICA RELEASE FACT CANDIDATE CONTEXT
 * =========================================================
 */

import {
    countSubjectMatches,
    getTitleIdentityText
} from "./releaseFactText.js";


import { extractDates } from "./releaseFactDates.js";


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


    // Do not associate this version with the date of a different release.
    // Accept only a date in the same row or a labelled immediately following line.
    const nextLine = ranges[ranges.indexOf(line) + 1];
    const matching = dates.filter(date => {
        const sameLine = line && date.index >= line.start && date.index <= line.end;
        const adjacent = nextLine
            && date.index >= nextLine.start && date.index <= nextLine.end
            && (
                /^(?:release\s+date|released(?:\s+on)?|дата\s+(?:выпуска|релиза))\s*[:—-]/iu.test(nextLine.text)
                || (nextLine.text.length < 40 && /^\s*(?:20\d{2}-\d{2}-\d{2}|[a-z]+\.?\s+\d|\d{1,2}\s+[\p{L}.]+\s+20\d{2})/iu.test(nextLine.text))
            );
        if (!sameLine && !adjacent) return false;
        const start = date.index < item.index
            ? date.index + date.raw.length : item.index + item.version.length;
        const end = date.index < item.index ? item.index : date.index;
        if (/\b\d{1,3}(?:\.\d{1,4}){1,3}\b/u.test(text.slice(start, end))) return false;
        return Math.abs(date.index - item.index) <= 220;
    });
    matching.sort((a,b) => Math.abs(a.index - item.index) - Math.abs(b.index - item.index));
    if (!matching.length) return null;
    const nearestDate = {
        ...matching[0],
        distance: Math.abs(matching[0].index - item.index)
    };


    return {

        lineText,

        nearby,

        lineSubjectMatches,

        nearbySubjectMatches,

        sourceSubjectMatches,

        nearestDate

    };

}
