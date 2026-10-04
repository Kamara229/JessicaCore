/*
 * =========================================================
 * JESSICA LEARNING SKILL BUILDER v4
 * =========================================================
 *
 * Центральный Builder
 * финального Experience Skill.
 *
 *
 * Flow:
 *
 * Proposed Experience
 *        ↓
 * Validation
 *        ↓
 * Identity / Version
 *        ↓
 * Knowledge Builder
 *        ↓
 * Learning Metrics Builder
 *        ↓
 * Metadata Builder
 *        ↓
 * Experience Skill
 *        ↓
 * Experience Storage
 *
 *
 * Ответственность:
 *
 * - координировать построение Skill;
 * - сформировать Identity;
 * - сформировать Version;
 * - объединить специализированные блоки;
 * - вернуть канонический Experience Skill.
 *
 *
 * НЕ:
 *
 * - принимает Learning Decision;
 * - вызывает AI;
 * - ищет версии;
 * - объединяет Candidate;
 * - работает с Supabase;
 * - принимает AUTO_APPROVE.
 *
 * =========================================================
 */


import {
    normalizeText,
    normalizeVersion
} from "./skill/skillUtils.js";


import {
    buildLearningSkillId
} from "./skill/skillId.js";


import {
    buildSkillKnowledge
} from "./skill/skillKnowledgeBuilder.js";


import {
    buildSkillLearning,
    buildInitialRuntimeStatistics
} from "./skill/skillLearningBuilder.js";


import {
    buildSkillMetadata
} from "./skill/skillMetadataBuilder.js";


import {
    validateSkillBuildInput,
    validateBuiltExperienceSkill
} from "./skill/skillValidation.js";





/*
 * =========================================================
 * PUBLIC ID BUILDER
 * =========================================================
 *
 * Re-export сохраняет совместимость:
 *
 * patternNormalizer.js
 * и другие существующие импорты
 * менять не требуется.
 *
 * =========================================================
 */


export {
    buildLearningSkillId
};





/*
 * =========================================================
 * MODE
 * =========================================================
 */


function normalizeMode(
    value
) {


    const mode =

        normalizeText(
            value
        )
        .toLowerCase();


    if(
        mode === "update"
    ){

        return "update";

    }


    return "create";

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

    confidence = null,

    metadata = {}

} = {}) {


    /*
     * =====================================================
     * 1. VALIDATE INPUT
     * =====================================================
     */


    validateSkillBuildInput(
        proposedExperience
    );



    /*
     * =====================================================
     * 2. IDENTITY
     * =====================================================
     */


    const name =

        normalizeText(
            proposedExperience.name
        );


    const id =

        normalizeText(
            skillId
        )

        ||

        normalizeText(
            proposedExperience.id
        )

        ||

        buildLearningSkillId(
            name
        );


    const normalizedName =

        buildLearningSkillId(
            name
        );



    /*
     * =====================================================
     * 3. VERSION
     * =====================================================
     */


    const normalizedVersion =

        normalizeVersion(
            version
        );


    const normalizedPreviousVersion =

        previousVersion !== null
        &&
        previousVersion !== undefined

            ? normalizeVersion(
                previousVersion
            )

            : null;


    const normalizedMode =

        normalizeMode(
            mode
        );



    /*
     * =====================================================
     * 4. KNOWLEDGE
     * =====================================================
     */


    const knowledge =

        buildSkillKnowledge(
            proposedExperience
        );



    /*
     * =====================================================
     * 5. LEARNING
     * =====================================================
     */


    const learning =

        buildSkillLearning({

            proposedExperience,

            confidence

        });



    /*
     * =====================================================
     * 6. METADATA
     * =====================================================
     */


    const skillMetadata =

        buildSkillMetadata({

            proposedExperience,

            metadata

        });



    /*
     * =====================================================
     * 7. BUILD
     * =====================================================
     */


    const skill = {


        /*
         * =================================================
         * IDENTITY
         * =================================================
         */


        id,


        name,


        normalizedName,


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


        source:

            normalizeText(
                proposedExperience.source
            )

            ||

            "execution-learning",



        /*
         * =================================================
         * VERSION
         * =================================================
         */


        version:

            normalizedVersion,


        previousVersion:

            normalizedPreviousVersion,


        mode:

            normalizedMode,



        /*
         * =================================================
         * STATUS
         * =================================================
         */


        enabled:

            true,


        /*
         * Legacy-compatible top-level confidence.
         *
         * Канонический источник:
         *
         * skill.learning.confidence
         */


        confidence:

            learning.confidence,



        /*
         * =================================================
         * KNOWLEDGE
         * =================================================
         */


        ...knowledge,



        /*
         * =================================================
         * LEARNING MEMORY
         * =================================================
         */


        learning,



        /*
         * =================================================
         * RUNTIME MEMORY
         * =================================================
         *
         * Runtime конкретной версии.
         *
         * Не путать с накопленным
         * skill.learning.
         *
         * =================================================
         */


        statistics:

            buildInitialRuntimeStatistics(),



        /*
         * =================================================
         * META
         * =================================================
         */


        metadata:

            skillMetadata,



        /*
         * =================================================
         * SYSTEM
         * =================================================
         */


        learnedAt:

            new Date()
                .toISOString(),


        builderVersion:

            "v4"

    };



    /*
     * =====================================================
     * 8. FINAL VALIDATION
     * =====================================================
     */


    validateBuiltExperienceSkill(
        skill
    );


    return skill;

}
