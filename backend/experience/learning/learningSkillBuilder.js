import {
    randomUUID
} from "node:crypto";


/*
 * =========================================================
 * JESSICA LEARNING SKILL BUILDER
 * =========================================================
 *
 * Превращает proposedExperience
 * из Learning Proposal
 * в полноценный Experience Skill.
 *
 *
 * Этот модуль отвечает только за:
 *
 * - Skill ID;
 * - version;
 * - enabled;
 * - confidence;
 * - нормализацию полей Skill.
 *
 *
 * Этот модуль НЕ:
 *
 * - сохраняет Skill;
 * - работает с Supabase;
 * - подтверждает Proposal;
 * - вызывает AI;
 * - определяет следующую версию сам.
 *
 * =========================================================
 */


/*
 * =========================================================
 * CYRILLIC TRANSLITERATION
 * =========================================================
 */


const TRANSLITERATION = {

    а: "a",
    б: "b",
    в: "v",
    г: "g",
    д: "d",
    е: "e",
    ё: "e",
    ж: "zh",
    з: "z",
    и: "i",
    й: "y",
    к: "k",
    л: "l",
    м: "m",
    н: "n",
    о: "o",
    п: "p",
    р: "r",
    с: "s",
    т: "t",
    у: "u",
    ф: "f",
    х: "h",
    ц: "ts",
    ч: "ch",
    ш: "sh",
    щ: "sch",
    ъ: "",
    ы: "y",
    ь: "",
    э: "e",
    ю: "yu",
    я: "ya"

};


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
    ).trim();

}


/*
 * =========================================================
 * NORMALIZE TEXT ARRAY
 * =========================================================
 */


function normalizeTextArray(
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
                normalizeText(
                    item
                )
        )
        .filter(
            Boolean
        );

}


/*
 * =========================================================
 * TRANSLITERATE
 * =========================================================
 */


function transliterate(
    value
) {

    return normalizeText(
        value
    )
        .toLowerCase()
        .split("")
        .map(
            character =>
                TRANSLITERATION[
                    character
                ] ??
                character
        )
        .join("");

}


/*
 * =========================================================
 * BUILD SKILL ID
 * =========================================================
 */


export function buildLearningSkillId(
    value
) {


    const normalized =
        transliterate(
            value
        )
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


    if (normalized) {

        return normalized;

    }


    /*
     * Редкий fallback:
     * если имя вообще нельзя
     * преобразовать в читаемый ID.
     */


    return `skill-${randomUUID()}`;

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

    confidence = 0.7

} = {}) {


    if (
        !proposedExperience ||
        typeof proposedExperience !== "object" ||
        Array.isArray(
            proposedExperience
        )
    ) {

        throw new Error(
            "proposedExperience не указан"
        );

    }


    const name =
        normalizeText(
            proposedExperience.name
        );


    if (!name) {

        throw new Error(
            "Название Skill не указано"
        );

    }


    /*
     * =====================================================
     * ID
     * =====================================================
     *
     * Если Approval передаст ID существующего
     * Skill — сохраняем его.
     *
     * Иначе создаём ID из названия.
     *
     * =====================================================
     */


    const id =
        normalizeText(
            skillId
        ) ||
        buildLearningSkillId(
            name
        );


    /*
     * =====================================================
     * VERSION
     * =====================================================
     */


    const normalizedVersion =
        Number(
            version
        );


    if (
        !Number.isInteger(
            normalizedVersion
        ) ||
        normalizedVersion < 1
    ) {

        throw new Error(
            "Некорректная версия Skill"
        );

    }


    /*
     * =====================================================
     * CONFIDENCE
     * =====================================================
     */


    const normalizedConfidence =
        Number(
            confidence
        );


    const safeConfidence =
        Number.isFinite(
            normalizedConfidence
        )
            ? Math.max(
                0,
                Math.min(
                    1,
                    normalizedConfidence
                )
            )
            : 0.7;


    /*
     * =====================================================
     * SKILL
     * =====================================================
     */


    return {

        id,

        name,

        description:
            normalizeText(
                proposedExperience.description
            ),

        version:
            normalizedVersion,

        enabled:
            true,

        confidence:
            safeConfidence,

        taskTypes:
            normalizeTextArray(
                proposedExperience.taskTypes
            ),

        keywords:
            normalizeTextArray(
                proposedExperience.keywords
            ),

        tags:
            normalizeTextArray(
                proposedExperience.tags
            ),

        strategy:
            normalizeTextArray(
                proposedExperience.strategy
            ),

        sourcePriority:
            normalizeTextArray(
                proposedExperience.sourcePriority
            ),

        validationRules:
            normalizeTextArray(
                proposedExperience.validationRules
            ),

        failurePatterns:
            normalizeTextArray(
                proposedExperience.failurePatterns
            ),

        successfulRuns:
            0,

        failedRuns:
            0,

        learnedAt:
            new Date()
                .toISOString(),

        learnedFrom:
            "user-correction"

    };

      }
