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
