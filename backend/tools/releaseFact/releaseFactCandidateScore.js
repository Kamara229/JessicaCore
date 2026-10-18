/*
 * =========================================================
 * JESSICA RELEASE FACT CANDIDATE SCORE
 * =========================================================
 */

import {
    RELEASE_SIGNAL
} from "./releaseFactConstants.js";


export function scoreCandidate({

    section,

    context

}) {

    let score = 0;


    if(
        section.preferred === true
    ){

        score += 24;

    }


    if(
        RELEASE_SIGNAL.test(
            context.lineText
        )
    ){

        score += 7;

    }


    if(
        RELEASE_SIGNAL.test(
            context.nearby
        )
    ){

        score += 4;

    }


    score +=

        Math.min(
            12,
            context.lineSubjectMatches * 8
        );


    score +=

        Math.min(
            8,
            context.nearbySubjectMatches * 4
        );


    score +=

        Math.min(
            5,
            context.sourceSubjectMatches * 3
        );


    if(
        context.nearestDate.distance <= 80
    ){

        score += 12;

    }
    else if(
        context.nearestDate.distance <= 220
    ){

        score += 7;

    }
    else{

        score += 3;

    }


    if(
        /(?:[·|]\s*20\d{2}-\d{2}-\d{2}|20\d{2}-\d{2}-\d{2}\s*[·|])/u
            .test(
                context.lineText
            )
    ){

        score += 8;

    }


    if(
        /\bnotes?\b/iu.test(
            context.lineText
        )
    ){

        score += 3;

    }


    return score;

}
