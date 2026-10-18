/*
 * =========================================================
 * JESSICA EXPERIENCE PATTERN MATCHER
 * =========================================================
 *
 * Поиск соответствия Execution Trace
 * известному Experience Pattern.
 *
 *
 * Flow:
 *
 * Execution Trace
 *        ↓
 * Pattern Matcher
 *        ↓
 * Matched Pattern
 *
 *
 * НЕ:
 *
 * - создаёт Skill;
 * - изменяет Experience;
 * - сохраняет данные.
 *
 * =========================================================
 */


import {
    EXPERIENCE_PATTERNS
} from "./experiencePatterns.js";





/*
 * =========================================================
 * NORMALIZE TEXT
 * =========================================================
 */


function normalizeText(
    value
) {

    return String(
        value || ""
    )
    .toLowerCase()
    .trim();

}








/*
 * =========================================================
 * MATCH PATTERN
 * =========================================================
 */


export function matchExperiencePattern(
    task
) {


    const text =
        normalizeText(
            task
        );



    if(
        !text
    ){

        return null;

    }





    let bestPattern =
        null;



    let bestScore =
        0;







    for(
        const pattern
        of EXPERIENCE_PATTERNS
    ){



        const matches =

            pattern.keywords.filter(

                keyword =>

                    text.includes(
                        normalizeText(
                            keyword
                        )
                    )

            )
            .length;





        if(
            matches > bestScore
        ){

            bestScore =
                matches;


            bestPattern =
                pattern;

        }


    }








    if(
        !bestPattern
    ){

        return null;

    }

    /*
     * Single generic keywords like "источник" or "сайт"
     * cannot establish a reusable procedure identity.
     *
     * Weak matches must enter dynamic Pattern Discovery,
     * not create a duplicate NEW_SKILL with a seed-pattern ID.
     */
    const candidateScore = bestScore / bestPattern.keywords.length;

    if (bestScore < 2 || candidateScore < 0.50) {
        return null;
    }







    return {


        pattern:

            bestPattern,



        matchedKeywords:

            bestPattern.keywords.filter(

                keyword =>

                    text.includes(
                        normalizeText(
                            keyword
                        )
                    )

            ),



        matchScore:

            Number(

                (

                    bestScore /

                    bestPattern.keywords.length

                )
                .toFixed(2)

            )


    };


}
