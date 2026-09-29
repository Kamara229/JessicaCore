/*
 * =========================================================
 * JESSICA EXPERIENCE MATCHER v0.5
 * =========================================================
 *
 * Match Task → Experience Skill
 *
 * Приоритет:
 *
 * 1. triggerPatterns
 * 2. keywords
 * 3. workflow
 * 4. successfulPatterns
 * 5. name
 * 6. description
 *
 * НЕ:
 *
 * - выбирает Skill;
 * - читает Storage;
 * - вызывает AI.
 *
 * =========================================================
 */


import {

    normalizeExperienceText,

    canonicalizeExperienceTokens,

    uniqueExperienceValues,

    normalizeExperienceStringArray

}
from "./experienceText.js";






const SCORE = {


    TRIGGER_PHRASE:
        0.50,


    TRIGGER_TERM:
        0.30,


    KEYWORD_PHRASE:
        0.35,


    KEYWORD_TERM:
        0.20,


    WORKFLOW:
        0.20,


    SUCCESS_PATTERN:
        0.15,


    NAME:
        0.15,


    DESCRIPTION:
        0.05

};







function emptyMatch(){

    return {

        confidence:0,

        matchedTerms:[],

        matchedPhrases:[],

        reasons:[]

    };

}








function containsPhrase(
    taskTokens,
    phraseTokens
){

    if(
        phraseTokens.length === 0
    ){

        return false;

    }



    for(
        let i = 0;

        i <= taskTokens.length - phraseTokens.length;

        i++
    ){

        let ok=true;


        for(
            let j=0;

            j<phraseTokens.length;

            j++
        ){

            if(
                taskTokens[i+j]
                !==
                phraseTokens[j]
            ){

                ok=false;
                break;

            }

        }


        if(ok)
            return true;

    }


    return false;

}









function matchPhrases(
    taskTokens,
    phrases,
    phraseScore,
    termScore
){

    let score=0;

    const terms=[];

    const matchedPhrases=[];



    for(
        const phrase of phrases
    ){

        const tokens =
            canonicalizeExperienceTokens(
                phrase
            );



        if(
            tokens.length===0
        ){

            continue;

        }



        if(
            tokens.length>1
            &&
            containsPhrase(
                taskTokens,
                tokens
            )
        ){

            score=Math.max(
                score,
                phraseScore
            );


            matchedPhrases.push(
                normalizeExperienceText(
                    phrase
                )
            );


            terms.push(
                ...tokens
            );


        }
        else if(
            tokens.length===1
            &&
            taskTokens.includes(
                tokens[0]
            )
        ){

            score=Math.max(
                score,
                termScore
            );


            terms.push(
                tokens[0]
            );

        }

    }



    return {

        score,

        terms:
            uniqueExperienceValues(
                terms
            ),

        phrases:
            uniqueExperienceValues(
                matchedPhrases
            )

    };

}









function overlap(
    taskSet,
    tokens
){

    const unique =
        uniqueExperienceValues(
            tokens
        );


    if(
        unique.length===0
    ){

        return {

            ratio:0,

            matched:[]

        };

    }


    const matched =
        unique.filter(
            x =>
            taskSet.has(x)
        );


    return {


        ratio:
            matched.length /
            unique.length,


        matched


    };


}









export function calculateExperienceMatch(
    task,
    experience
){

    const taskTokens =
        canonicalizeExperienceTokens(
            task
        );


    if(
        taskTokens.length===0
    ){

        return emptyMatch();

    }



    const taskSet =
        new Set(
            taskTokens
        );








    const triggerMatch =
        matchPhrases(

            taskTokens,

            normalizeExperienceStringArray(
                experience?.triggerPatterns
            ),

            SCORE.TRIGGER_PHRASE,

            SCORE.TRIGGER_TERM

        );






    const keywordMatch =
        matchPhrases(

            taskTokens,

            normalizeExperienceStringArray(
                experience?.keywords
            ),

            SCORE.KEYWORD_PHRASE,

            SCORE.KEYWORD_TERM

        );






    const workflowMatch =
        overlap(

            taskSet,

            canonicalizeExperienceTokens(

                (
                    experience?.workflow || []
                )
                .join(" ")

            )

        );







    const successMatch =
        overlap(

            taskSet,

            canonicalizeExperienceTokens(

                (
                    experience?.successfulPatterns || []
                )
                .join(" ")

            )

        );







    const nameMatch =
        overlap(

            taskSet,

            canonicalizeExperienceTokens(
                experience?.name
            )

        );







    const descriptionMatch =
        overlap(

            taskSet,

            canonicalizeExperienceTokens(
                experience?.description
            )

        );








    let confidence=0;


    const reasons=[];



    if(triggerMatch.score){

        confidence +=
            triggerMatch.score;


        reasons.push(
            "trigger-pattern-match"
        );

    }



    confidence +=
        keywordMatch.score;


    if(keywordMatch.score){

        reasons.push(
            "keyword-match"
        );

    }



    confidence +=
        workflowMatch.ratio *
        SCORE.WORKFLOW;



    if(workflowMatch.ratio){

        reasons.push(
            "workflow-match"
        );

    }



    confidence +=
        successMatch.ratio *
        SCORE.SUCCESS_PATTERN;



    confidence +=
        nameMatch.ratio *
        SCORE.NAME;



    confidence +=
        descriptionMatch.ratio *
        SCORE.DESCRIPTION;



    confidence =
        Math.min(
            1,
            confidence
        );







    return {


        confidence,


        matchedTerms:

            uniqueExperienceValues([

                ...triggerMatch.terms,

                ...keywordMatch.terms,

                ...workflowMatch.matched,

                ...successMatch.matched,

                ...nameMatch.matched,

                ...descriptionMatch.matched

            ]),



        matchedPhrases:

            uniqueExperienceValues([

                ...triggerMatch.phrases,

                ...keywordMatch.phrases

            ]),



        reasons


    };


}
