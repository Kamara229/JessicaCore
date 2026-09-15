/*
 * =========================================================
 * JESSICA EXPERIENCE CONTEXT v0.2
 * =========================================================
 *
 * Преобразует найденный Experience Skill
 * в безопасный контекст для Planner.
 *
 *
 * Вход:
 *
 * ExperienceResult
 *
 *
 * Выход:
 *
 * {
 *     experience: {...},
 *     metadata: {...}
 * }
 *
 *
 * Этот модуль НЕ:
 *
 * - ищет Experience;
 * - читает Storage;
 * - сохраняет Skills;
 * - обучает Jessica;
 * - вызывает Planner;
 * - выполняет задачи.
 *
 *
 * Задача:
 *
 * Experience Result
 *        ↓
 * Planning-compatible context
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



/*
 * =========================================================
 * NORMALIZE STRING ARRAY
 * =========================================================
 */


function normalizeStringArray(
    value
) {

    if (
        !Array.isArray(
            value
        )
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
        );

}



/*
 * =========================================================
 * LIMIT ARRAY
 * =========================================================
 *
 * Защита Planner Context
 * от слишком больших Skill.
 *
 * =========================================================
 */


function limitArray(
    value,
    limit = MAX_ARRAY_ITEMS
) {

    return normalizeStringArray(
        value
    )
        .slice(
            0,
            limit
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


    return Number.isFinite(
        number
    )
        ? number
        : fallback;

}



/*
 * =========================================================
 * BUILD EXPERIENCE CONTEXT
 * =========================================================
 */


export function buildExperienceContext(
    experienceResult
) {


    /*
     * =====================================================
     * NO EXPERIENCE
     * =====================================================
     */


    if (
        !experienceResult ||
        experienceResult.found !== true ||
        !experienceResult.experience ||
        typeof experienceResult.experience !== "object"
    ) {

        return {

            experience:
                null,

            metadata:
                {}

        };

    }



    const sourceExperience =
        experienceResult.experience;



    /*
     * =====================================================
     * SAFE EXPERIENCE
     * =====================================================
     *
     * Передаём Planner только
     * полезные данные Skill.
     *
     * Служебные поля Storage
     * и история версий сюда
     * не попадают.
     *
     * =====================================================
     */


    const experience = {


        skillId:
            sourceExperience.id ||
            sourceExperience.skillId ||
            null,


        name:
            String(
                sourceExperience.name || ""
            )
            .trim(),


        description:
            String(
                sourceExperience.description || ""
            )
            .trim(),


        version:
            normalizeNumber(
                sourceExperience.version,
                1
            ),



        /*
         * Уверенность самого Skill.
         *
         * Например:
         *
         * насколько качественно
         * Jessica его получила.
         */

        skillConfidence:
            normalizeNumber(
                sourceExperience.confidence,
                0
            ),



        /*
         * Уверенность поиска.
         *
         * Насколько этот Skill
         * подходит текущей задаче.
         */

        matchConfidence:
            normalizeNumber(
                experienceResult.confidence,
                0
            ),



        keywords:
            limitArray(
                sourceExperience.keywords
            ),


        tags:
            limitArray(
                sourceExperience.tags
            ),


        strategy:
            limitArray(
                sourceExperience.strategy
            ),


        sourcePriority:
            limitArray(
                sourceExperience.sourcePriority
            ),


        validationRules:
            limitArray(
                sourceExperience.validationRules
            ),


        failurePatterns:
            limitArray(
                sourceExperience.failurePatterns
            )

    };



    /*
     * =====================================================
     * METADATA
     * =====================================================
     *
     * Служебная информация:
     *
     * - как найден Experience;
     * - какой Skill использован;
     * - какая версия;
     * - когда найден.
     *
     * Используется:
     *
     * - логи;
     * - аналитика;
     * - обучение;
     * - Earnings.
     *
     * =====================================================
     */


    const metadata = {


        experienceFound:
            true,


        experienceSource:
            experienceResult.source ||
            "unknown",


        experienceMatchConfidence:
            experience.matchConfidence,


        skillConfidence:
            experience.skillConfidence,


        skillId:
            experience.skillId,


        skillName:
            experience.name,


        skillVersion:
            experience.version,


        matchedAt:
            new Date()
                .toISOString()

    };



    /*
     * =====================================================
     * RESULT
     * =====================================================
     */


    return {

        experience,

        metadata

    };


}
