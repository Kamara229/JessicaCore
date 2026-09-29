import {
    randomUUID
} from "node:crypto";



/*
 * =========================================================
 * JESSICA LEARNING SKILL BUILDER v3
 * =========================================================
 *
 * Создание Experience Skill.
 *
 *
 * Flow:
 *
 * Learning Proposal
 *        ↓
 * Autonomy Approval
 *        ↓
 * Skill Builder
 *        ↓
 * Experience Storage
 *
 *
 * НЕ:
 *
 * - принимает решение обучения;
 * - работает с БД;
 * - ищет версии;
 * - вызывает AI.
 *
 * =========================================================
 */





const TRANSLITERATION = {

    а:"a",
    б:"b",
    в:"v",
    г:"g",
    д:"d",
    е:"e",
    ё:"e",
    ж:"zh",
    з:"z",
    и:"i",
    й:"y",
    к:"k",
    л:"l",
    м:"m",
    н:"n",
    о:"o",
    п:"p",
    р:"r",
    с:"s",
    т:"t",
    у:"u",
    ф:"f",
    х:"h",
    ц:"ts",
    ч:"ch",
    ш:"sh",
    щ:"sch",
    ъ:"",
    ы:"y",
    ь:"",
    э:"e",
    ю:"yu",
    я:"ya"

};









function normalizeText(
    value
){

    return String(
        value || ""
    )
    .trim();

}









function normalizeArray(
    value
){

    if(
        !Array.isArray(value)
    ){

        return [];

    }


    return value

        .map(
            item =>
                normalizeText(item)
        )

        .filter(
            Boolean
        );

}









function normalizeConfidence(
    value
){

    const number =
        Number(value);


    if(
        !Number.isFinite(number)
    ){

        return 0.7;

    }


    return Math.max(
        0,
        Math.min(
            1,
            number
        )
    );

}









function normalizeVersion(
    value
){

    const version =
        Number(value);


    if(
        !Number.isInteger(version)
        ||
        version < 1
    ){

        return 1;

    }


    return version;

}









/*
 * =========================================================
 * BUILD SKILL ID
 * =========================================================
 */


export function buildLearningSkillId(
    value
){

    const normalized =

        normalizeText(
            value
        )

        .toLowerCase()

        .split("")

        .map(
            char =>
                TRANSLITERATION[char]
                ??
                char
        )

        .join("")

        .replace(
            /[^a-z0-9]+/g,
            "-"
        )

        .replace(
            /^-+|-+$/g,
            ""
        )

        .slice(
            0,
            80
        );



    if(
        normalized
    ){

        return normalized;

    }



    return (

        "skill-"

        +

        randomUUID()
            .slice(
                0,
                8
            )

    );

}









/*
 * =========================================================
 * BUILD EXPERIENCE SKILL
 * =========================================================
 */


export function buildExperienceSkill({

    proposedExperience,

    skillId = "",

    version = 1,

    previousVersion = null,

    mode = "create",

    confidence = 0.7,

    metadata = {}

} = {}) {



    if(
        !proposedExperience
        ||
        typeof proposedExperience !== "object"
    ){

        throw new Error(
            "proposedExperience отсутствует"
        );

    }





    const name =

        normalizeText(
            proposedExperience.name
        );



    if(
        !name
    ){

        throw new Error(
            "Название Skill отсутствует"
        );

    }







    const id =

        normalizeText(
            skillId
        )

        ||

        buildLearningSkillId(
            name
        );









    return {





        /*
         * IDENTITY
         */


        id,


        name,


        normalizedName:

            buildLearningSkillId(
                name
            ),



        category:

            normalizeText(
                proposedExperience.category
            )
            ||
            "general",



        description:

            normalizeText(
                proposedExperience.description
            ),







        /*
         * VERSION
         */


        version:

            normalizeVersion(
                version
            ),



        previousVersion:

            previousVersion
            ?
            normalizeVersion(
                previousVersion
            )
            :
            null,



        mode,









        /*
         * STATUS
         */


        enabled:true,



        confidence:

            normalizeConfidence(
                confidence
            ),







        /*
         * KNOWLEDGE
         */


        workflow:

            Array.isArray(
                proposedExperience.workflow
            )

            ?

            proposedExperience.workflow

            :

            [],



        triggerPatterns:

            normalizeArray(
                proposedExperience.triggerPatterns
            ),



        keywords:

            normalizeArray(
                proposedExperience.keywords
            ),



        tags:

            normalizeArray(
                proposedExperience.tags
            ),



        validationRules:

            normalizeArray(
                proposedExperience.validationRules
            ),



        constraints:

            normalizeArray(
                proposedExperience.constraints
            ),



        strategy:

            normalizeArray(
                proposedExperience.strategy
            ),



        sourcePriority:

            normalizeArray(
                proposedExperience.sourcePriority
            ),



        successfulPatterns:

            normalizeArray(
                proposedExperience.successfulPatterns
            ),



        failurePatterns:

            normalizeArray(
                proposedExperience.failurePatterns
            ),



        avoidPatterns:

            normalizeArray(
                proposedExperience.avoidPatterns
            ),



        examples:

            Array.isArray(
                proposedExperience.examples
            )

            ?

            proposedExperience.examples

            :

            [],







        /*
         * RUNTIME MEMORY
         */


        statistics:{


            successfulRuns:0,


            failedRuns:0,


            lastUsedAt:null,


            lastResult:null


        },









        /*
         * LEARNING META
         */


        metadata:{


            proposalId:

                metadata.proposalId ||
                null,


            queueItemId:

                metadata.queueItemId ||
                null,


            learnedFrom:

                metadata.learnedFrom ||
                "learning_pipeline",



            learning:


                metadata.learning
                ||
                null,



            createdBy:

                metadata.createdBy
                ||
                "jessica-learning",



            sourceExperience:

                metadata.sourceExperience
                ||
                null


        },







        learnedAt:

            new Date()
                .toISOString(),



        builderVersion:

            "v3"


    };

}
