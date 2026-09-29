/*
 * =========================================================
 * JESSICA EXPERIENCE CANDIDATE BUILDER
 * =========================================================
 *
 * Создаёт Learning Candidate
 * из Execution Trace и найденного Pattern.
 *
 *
 * Flow:
 *
 * Pattern Matcher
 *        +
 * Confidence Calculator
 *        ↓
 * Candidate Builder
 *        ↓
 * Skill Candidate
 *
 *
 * НЕ:
 *
 * - сохраняет Skill;
 * - вызывает AI;
 * - принимает решение обучения.
 *
 * =========================================================
 */



/*
 * =========================================================
 * NORMALIZE
 * =========================================================
 */


function normalizeArray(
    value
) {


    if(
        !Array.isArray(value)
    ){

        return [];

    }



    return value;

}









/*
 * =========================================================
 * BUILD EXAMPLES
 * =========================================================
 */


function buildExamples(
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
 * BUILD NEW SKILL CANDIDATE
 * =========================================================
 */


export function buildNewSkillCandidate({

    pattern,

    trace,

    confidence,

    maturity,

    occurrences

}) {



    if(
        !pattern
    ){

        return null;

    }





    const examples =

        buildExamples(
            trace
        );







    const successCount =

        examples.filter(

            item =>

                item?.success !== false

        )
        .length;









    return {


        skillId:

            pattern.id,



        name:

            pattern.name,



        category:

            pattern.category,



        description:

            pattern.description,



        workflow:

            normalizeArray(
                pattern.workflow
            ),



        validationRules:

            normalizeArray(
                pattern.validationRules
            ),



        triggerPatterns:

            normalizeArray(
                pattern.keywords
            ),



        constraints:

            [],



        examples,



        occurrences,



        successCount,



        maturity,



        confidence,



        source:

            "execution-trace"


    };


}









/*
 * =========================================================
 * BUILD SKILL IMPROVEMENT CANDIDATE
 * =========================================================
 */


export function buildSkillImprovementCandidate({

    skills,

    trace,

    confidence,

    maturity,

    occurrences

}) {


    return {


        skills:

            Array.isArray(skills)

                ? skills

                : [],



        examples:

            buildExamples(
                trace
            ),



        occurrences,



        maturity,



        confidence,



        source:

            "execution-trace"


    };


}
