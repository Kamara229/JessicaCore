import {
    randomUUID
} from "node:crypto";


/*
 * =========================================================
 * JESSICA LEARNING SKILL BUILDER
 * =========================================================
 *
 * Создаёт Experience Skill из Learning Proposal.
 *
 *
 * Flow:
 *
 * Learning Proposal
 *        ↓
 * Skill Builder
 *        ↓
 * Experience Skill Version
 *        ↓
 * Experience Storage
 *
 *
 * Ответственность:
 *
 * - нормализация Skill;
 * - создание ID;
 * - создание версии;
 * - сохранение метаданных обучения.
 *
 *
 * НЕ:
 *
 * - сохраняет в БД;
 * - определяет Approval;
 * - ищет историю версий;
 * - вызывает AI.
 *
 * =========================================================
 */





/*
 * =========================================================
 * TRANSLITERATION
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





/*
 * =========================================================
 * NORMALIZE
 * =========================================================
 */


function normalizeText(
    value
) {

    return String(
        value || ""
    )
    .trim();

}





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
                normalizeText(item)
        )
        .filter(Boolean);

}





function normalizeObject(
    value
) {

    if (
        !value ||
        typeof value !== "object" ||
        Array.isArray(value)
    ) {

        return {};

    }


    return {
        ...value
    };

}







/*
 * =========================================================
 * ID BUILDER
 * =========================================================
 */


export function buildLearningSkillId(
    value
) {


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



    if (
        normalized
    ) {

        return normalized;

    }



    return (

        "generated-skill-"

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
 * VERSION
 * =========================================================
 */


function normalizeVersion(
    value
) {


    const version =
        Number(value);



    if (
        !Number.isInteger(version)
        ||
        version < 1
    ) {

        return 1;

    }


    return version;

}






/*
 * =========================================================
 * CONFIDENCE
 * =========================================================
 */


function normalizeConfidence(
    value
) {


    const confidence =
        Number(value);



    if (
        !Number.isFinite(confidence)
    ) {

        return 0.7;

    }



    return Math.max(

        0,

        Math.min(

            1,

            confidence

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



    if (
        !proposedExperience ||
        typeof proposedExperience !== "object"
    ) {

        throw new Error(
            "proposedExperience отсутствует"
        );

    }





    const name =
        normalizeText(
            proposedExperience.name
        );



    if (
        !name
    ) {

        throw new Error(
            "Название Skill отсутствует"
        );

    }





    const id =

        normalizeText(skillId)

        ||

        buildLearningSkillId(
            name
        );







    return {


        /*
         * Identity
         */


        id,


        name,


        normalizedName:
            buildLearningSkillId(
                name
            ),




        description:
            normalizeText(
                proposedExperience.description
            ),




        /*
         * Versioning
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
         * Status
         */


        enabled:
            true,



        confidence:
            normalizeConfidence(
                confidence
            ),





        /*
         * Knowledge
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



        examples:
            Array.isArray(
                proposedExperience.examples
            )
            ?
            proposedExperience.examples
            :
            [],



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



        validationRules:
            normalizeArray(
                proposedExperience.validationRules
            ),



        failurePatterns:
            normalizeArray(
                proposedExperience.failurePatterns
            ),



        successfulPatterns:
            normalizeArray(
                proposedExperience.successfulPatterns
            ),



        avoidPatterns:
            normalizeArray(
                proposedExperience.avoidPatterns
            ),






        /*
         * Statistics
         */


        successfulRuns:
            0,


        failedRuns:
            0,





        /*
         * Learning metadata
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
                "learning_pipeline"

        },



        learnedAt:
            new Date()
                .toISOString()


    };

}
