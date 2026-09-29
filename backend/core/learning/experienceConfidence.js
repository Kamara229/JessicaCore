/*
 * =========================================================
 * JESSICA EXPERIENCE CONFIDENCE
 * =========================================================
 *
 * Расчёт качества и зрелости опыта.
 *
 *
 * Используется:
 *
 * Experience Analyzer
 *        ↓
 * Confidence Calculator
 *
 *
 * НЕ:
 *
 * - ищет паттерны;
 * - создаёт Skill;
 * - сохраняет Experience.
 *
 * =========================================================
 */







/*
 * =========================================================
 * NORMALIZE NUMBER
 * =========================================================
 */


function normalizeNumber(
    value
) {


    const number =
        Number(value);



    if(
        Number.isNaN(number)
    ){

        return 0;

    }


    return number;

}









/*
 * =========================================================
 * OCCURRENCES
 * =========================================================
 */


export function getOccurrences(
    trace
) {


    return Math.max(

        normalizeNumber(

            trace?.experienceOccurrences

        )
        ||

        normalizeNumber(

            trace?.stats?.occurrences

        )
        ||

        1,

        1

    );


}









/*
 * =========================================================
 * EXAMPLES
 * =========================================================
 */


export function getExamples(
    trace
) {


    if(
        Array.isArray(
            trace?.examples
        )
    ){

        return trace.examples;

    }



    return [

        {

            task:

                trace?.task || "",



            result:

                trace?.result || "",



            success:

                true

        }

    ];

}









/*
 * =========================================================
 * SUCCESS RATE
 * =========================================================
 */


export function calculateSuccessRate(
    examples = []
) {


    if(
        !Array.isArray(examples)
        ||
        examples.length === 0
    ){

        return 0;

    }





    const successCount =

        examples.filter(

            item =>

                item?.success !== false

        )
        .length;





    return Number(

        (

            successCount /

            examples.length

        )
        .toFixed(2)

    );


}









/*
 * =========================================================
 * CONFIDENCE
 * =========================================================
 *
 * Формула:
 *
 * Match quality     30%
 * Success rate      40%
 * Repetition        30%
 *
 * =========================================================
 */


export function calculateExperienceConfidence({

    matchScore = 0,

    successRate = 0,

    occurrences = 1

} = {}) {



    const repeatScore =

        Math.min(

            occurrences / 5,

            1

        );





    return Number(

        (

            matchScore * 0.3 +

            successRate * 0.4 +

            repeatScore * 0.3

        )
        .toFixed(2)

    );


}









/*
 * =========================================================
 * MATURITY
 * =========================================================
 */


export function calculateExperienceMaturity(
    occurrences
) {


    const count =
        normalizeNumber(
            occurrences
        );



    if(
        count >= 5
    ){

        return "READY";

    }



    if(
        count >= 3
    ){

        return "LEARNING";

    }



    return "NEW";


}
