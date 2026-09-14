/*
 * =========================================================
 * JESSICA EXPERIENCE CONTEXT
 * =========================================================
 *
 * Преобразует найденный Skill
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
 * - читает базу;
 * - сохраняет Skills;
 * - обучает Jessica;
 * - вызывает Planner;
 * - выполняет задачу.
 *
 *
 * Его единственная задача:
 *
 * Experience Result
 *      ↓
 * Planning-compatible context
 *
 * =========================================================
 */


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
                ).trim()
        )
        .filter(
            Boolean
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
     * Передаём Planner только те поля,
     * которые реально нужны для планирования.
     *
     * Внутренние данные хранения,
     * служебные поля и будущая история
     * обучения сюда не попадают.
     *
     * =====================================================
     */


    const experience = {

        skillId:
            sourceExperience.id ||
            sourceExperience.skillId ||
            null,

        name:
            sourceExperience.name ||
            "",

        version:
            normalizeNumber(
                sourceExperience.version,
                1
            ),

        confidence:
            normalizeNumber(
                sourceExperience.confidence,
                0
            ),

        strategy:
            normalizeStringArray(
                sourceExperience.strategy
            ),

        sourcePriority:
            normalizeStringArray(
                sourceExperience.sourcePriority
            ),

        validationRules:
            normalizeStringArray(
                sourceExperience.validationRules
            ),

        failurePatterns:
            normalizeStringArray(
                sourceExperience.failurePatterns
            )

    };



    /*
     * =====================================================
     * METADATA
     * =====================================================
     *
     * Отдельно сохраняем информацию
     * о том, как Experience был найден.
     *
     * Это пригодится:
     *
     * - для логов;
     * - для обучения;
     * - для оценки качества поиска;
     * - для Earnings;
     * - для последующего анализа.
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
            normalizeNumber(
                experienceResult.confidence,
                0
            ),

        skillId:
            experience.skillId,

        skillVersion:
            experience.version

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
