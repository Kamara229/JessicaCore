/*
 * =========================================================
 * JESSICA RELEASE FACT CANDIDATES
 * =========================================================
 */

import {
    collectVersionMatches,
    compareVersions,
    isPrereleaseBound
} from "./releaseFactVersions.js";


import {
    buildCandidateEnvironment,
    buildCandidateContext
} from "./releaseFactCandidateContext.js";


import {
    scoreCandidate
} from "./releaseFactCandidateScore.js";


export function buildCandidates({

    section,

    sourceTitle,

    sourceUrl,

    subjectTokens

}) {

    const environment =

        buildCandidateEnvironment({

            section,

            sourceTitle,

            subjectTokens

        });


    const candidates = [];


    for(
        const item
        of collectVersionMatches(
            section.text
        )
    ){

        if(
            isPrereleaseBound({

                text:
                    section.text,

                version:
                    item.version,

                index:
                    item.index

            })
        ){

            continue;

        }


        const context =

            buildCandidateContext({

                environment,

                item,

                subjectTokens

            });


        if(!context){

            continue;

        }

        // Planned and pre-release rows never represent a stable release.
        if (/\b(?:pre-release|prerelease|alpha|beta|preview|planned|release candidate)\b/iu.test(context.lineText)) {
            continue;
        }


        if(
            section.preferred !== true
            &&
            subjectTokens.length > 0
            &&
            context.lineSubjectMatches === 0
            &&
            context.nearbySubjectMatches === 0
            &&
            context.sourceSubjectMatches === 0
        ){

            continue;

        }


        const score =

            scoreCandidate({

                section,

                context

            });


        candidates.push({

            version:
                item.version,

            kind:
                item.kind,

            index:
                item.index,

            score,

            releaseDate:
                context.nearestDate.value,

            rawDate:
                context.nearestDate.raw,

            dateIndex:
                context.nearestDate.index,

            line:
                context.lineText,

            subjectMatches:

                context.lineSubjectMatches
                +
                context.nearbySubjectMatches
                +
                context.sourceSubjectMatches,

            sourceUrl

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

                // Highest stable version takes precedence over heuristic score.
                const byVersion = compareVersions(right.version, left.version);
                if (byVersion !== 0) return byVersion;
                const precision = right.version.split(".").length - left.version.split(".").length;
                if (precision !== 0) return precision;
                if (right.score !== left.score) return right.score - left.score;
                return left.index - right.index;

            }
        )[0]

        ||

        null;

}
