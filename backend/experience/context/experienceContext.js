/*
 * =========================================================
 * JESSICA EXPERIENCE CONTEXT v3
 * =========================================================
 *
 * Преобразует Experience Skill
 * в Planning Context.
 *
 * Search
 *    ↓
 * Context
 *    ↓
 * Planner
 *
 * НЕ:
 *
 * - ищет Skill;
 * - сохраняет Skill;
 * - обучает.
 *
 * =========================================================
 */



const MAX_ARRAY_ITEMS = 10;

const MAX_WORKFLOW_ITEMS = 20;

const MAX_RANKING_ITEMS = 5;







function normalizeArray(
    value,
    limit = MAX_ARRAY_ITEMS
){

    if(
        !Array.isArray(value)
    ){

        return [];

    }


    return value

        .map(
            item =>
                String(item || "")
                .trim()
        )

        .filter(Boolean)

        .slice(
            0,
            limit
        );

}







function normalizeNumber(
    value,
    fallback = 0
){

    const number =
        Number(value);


    return Number.isFinite(number)
        ?
        number
        :
        fallback;

}







function buildMatchMetadata(
    result
){

    return {


        matchedTerms:

            normalizeArray(
                result?.matchedTerms
            ),



        matchedPhrases:

            normalizeArray(
                result?.matchedPhrases
            ),



        matchReasons:

            normalizeArray(
                result?.matchReasons
            ),



        ranking:

            Array.isArray(
                result?.ranking
            )

            ?

            result.ranking
                .slice(
                    0,
                    MAX_RANKING_ITEMS
                )

            :

            []

    };

}







function buildPlannerHints(
    skill
){

    const hints = [];


    hints.push(
        "Используй Experience как проверенную рекомендацию, а не как жёсткий сценарий."
    );



    if(
        skill.validationRules?.length
    ){

        hints.push(
            "Применяй правила проверки Experience."
        );

    }



    if(
        skill.failurePatterns?.length
    ){

        hints.push(
            "Избегай известных ошибок предыдущих выполнений."
        );

    }



    if(
        skill.successfulPatterns?.length
    ){

        hints.push(
            "Используй подтверждённые успешные подходы."
        );

    }


    return hints;

}








export function buildExperienceContext(
    experienceResult
){

    if(

        !experienceResult

        ||

        experienceResult.found !== true

        ||

        !experienceResult.experience

    ){

        return {

            experience:null,

            sourceRules:[],

            constraints:[],

            instructions:[],

            plannerHints:[],

            metadata:{
                experienceFound:false
            }

        };

    }






    const skill =
        experienceResult.experience;








    const experience = {


        id:

            skill.id ||
            null,



        name:

            skill.name || "",



        description:

            skill.description || "",



        version:

            normalizeNumber(
                skill.version,
                1
            ),



        confidence:

            normalizeNumber(
                skill.confidence
            ),



        workflow:

            normalizeArray(
                skill.workflow,
                MAX_WORKFLOW_ITEMS
            ),



        keywords:

            normalizeArray(
                skill.keywords
            ),



        strategy:

            normalizeArray(
                skill.strategy
            ),



        validationRules:

            normalizeArray(
                skill.validationRules
            ),



        constraints:

            normalizeArray(
                skill.constraints
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
            ),




        usage:

        {

            successfulRuns:

                normalizeNumber(
                    skill.usage?.successfulRuns
                ),


            failedRuns:

                normalizeNumber(
                    skill.usage?.failedRuns
                ),


            lastUsedAt:

                skill.usage?.lastUsedAt || null

        }



    };







    return {


        experience,



        sourceRules:

            normalizeArray(
                skill.sourcePriority
            ),



        constraints:

            normalizeArray(
                skill.constraints
            ),



        instructions:

            normalizeArray(
                skill.instructions
            ),



        plannerHints:

            buildPlannerHints(
                skill
            ),



        metadata:


        {


            experienceFound:true,


            source:

                experienceResult.source ||
                "experience-search",



            skillId:

                skill.id,



            skillVersion:

                skill.version,



            skillConfidence:

                skill.confidence,



            matchConfidence:

                normalizeNumber(
                    experienceResult.confidence
                ),



            learningMode:

                skill.metadata?.learningMode ||
                null,



            createdBy:

                skill.metadata?.createdBy ||
                null,



            matchedAt:

                new Date()
                    .toISOString(),



            ...buildMatchMetadata(
                experienceResult
            )


        }


    };


}
