/*
 * =========================================================
 * JESSICA EXPERIENCE CONTEXT
 * =========================================================
 *
 * Преобразует найденный Experience Skill
 * в PlanningContext для Planner.
 *
 *
 * Flow:
 *
 * Experience Search Result
 *          ↓
 * Context Builder
 *          ↓
 * PlanningContext
 *          ↓
 * Planner
 *
 *
 * НЕ:
 *
 * - ищет Experience;
 * - читает Storage;
 * - сохраняет Skills;
 * - обучает Jessica;
 * - вызывает AI.
 *
 * =========================================================
 */





/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


const MAX_ARRAY_ITEMS =
    10;


const MAX_RANKING_ITEMS =
    5;






/*
 * =========================================================
 * NORMALIZE ARRAY
 * =========================================================
 */


function normalizeArray(
    value
) {


    if (
        !Array.isArray(value)
    ) {

        return [];

    }


    return value

        .map(
            item =>
                String(
                    item || ""
                )
                .trim()
        )

        .filter(
            Boolean
        )

        .slice(
            0,
            MAX_ARRAY_ITEMS
        );

}






/*
 * =========================================================
 * NORMALIZE NUMBER
 * =========================================================
 */


function normalizeNumber(
    value,
    fallback = 0
) {


    const number =
        Number(
            value
        );


    return Number.isFinite(number)
        ? number
        : fallback;

}






/*
 * =========================================================
 * BUILD MATCH INFORMATION
 * =========================================================
 */


function buildMatchMetadata(
    experienceResult
) {


    return {


        matchedTerms:

            normalizeArray(
                experienceResult.matchedTerms
            ),



        matchedPhrases:

            normalizeArray(
                experienceResult.matchedPhrases
            ),



        matchReasons:

            normalizeArray(
                experienceResult.matchReasons
            ),



        ranking:

            Array.isArray(
                experienceResult.ranking
            )

            ?

            experienceResult.ranking
                .slice(
                    0,
                    MAX_RANKING_ITEMS
                )

            :

            []

    };


}






/*
 * =========================================================
 * BUILD EXPERIENCE CONTEXT
 * =========================================================
 */


export function buildExperienceContext(
    experienceResult
) {



    if (

        !experienceResult

        ||

        experienceResult.found !== true

        ||

        !experienceResult.experience

        ||

        typeof experienceResult.experience !== "object"

    ) {


        return {


            experience:
                null,


            sourceRules:
                [],


            constraints:
                [],


            instructions:
                [],


            plannerHints:
                [],


            metadata:
                {

                    experienceFound:
                        false

                }

        };


    }






    const skill =
        experienceResult.experience;







    /*
     * =====================================================
     * EXPERIENCE CONTEXT
     * =====================================================
     */


    const experience = {


        skillId:

            skill.id ||
            skill.skillId ||
            null,



        name:

            String(
                skill.name || ""
            )
            .trim(),



        description:

            String(
                skill.description || ""
            )
            .trim(),



        version:

            normalizeNumber(
                skill.version,
                1
            ),



        skillConfidence:

            normalizeNumber(
                skill.confidence,
                0
            ),



        matchConfidence:

            normalizeNumber(
                experienceResult.confidence,
                0
            ),



        keywords:

            normalizeArray(
                skill.keywords
            ),



        tags:

            normalizeArray(
                skill.tags
            ),



        strategy:

            normalizeArray(
                skill.strategy
            ),



        sourcePriority:

            normalizeArray(
                skill.sourcePriority
            ),



        validationRules:

            normalizeArray(
                skill.validationRules
            ),



        failurePatterns:

            normalizeArray(
                skill.failurePatterns
            ),



        successfulPatterns:

            normalizeArray(
                skill.successfulPatterns
            ),



        avoidPatterns:

            normalizeArray(
                skill.avoidPatterns
            )

    };







    /*
     * =====================================================
     * PLANNER RULES
     * =====================================================
     */


    const sourceRules =

        normalizeArray(
            skill.sourcePriority
        );



    const constraints =

        normalizeArray(
            skill.constraints
        );



    const instructions =

        normalizeArray(
            skill.instructions
        );







    /*
     * =====================================================
     * HINTS
     * =====================================================
     */


    const plannerHints = [

        "Используй Experience как рекомендацию, а не как жёсткий сценарий.",

        "Проверяй актуальность результата.",

        "Избегай известных ошибок Skill."

    ];







    /*
     * =====================================================
     * METADATA
     * =====================================================
     */


    const metadata = {


        experienceFound:
            true,



        experienceSource:

            experienceResult.source ||
            "experience-search",



        skillId:
            experience.skillId,



        skillVersion:
            experience.version,



        skillName:
            experience.name,



        skillConfidence:
            experience.skillConfidence,



        matchConfidence:
            experience.matchConfidence,



        matchedAt:
            new Date()
                .toISOString(),



        ...buildMatchMetadata(
            experienceResult
        )

    };







    return {


        experience,


        sourceRules,


        constraints,


        instructions,


        plannerHints,


        metadata


    };


}
