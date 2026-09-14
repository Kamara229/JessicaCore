import {
    getSupabaseClient
} from "../../storage/supabaseClient.js";


/*
 * =========================================================
 * JESSICA EXPERIENCE HISTORY STORE
 * =========================================================
 *
 * Работает только с историей версий Skills.
 *
 *
 * Таблица:
 *
 * jessica_experience_skill_versions
 *
 *
 * Отвечает за:
 *
 * - сохранение версии Skill;
 * - получение истории Skill;
 * - получение конкретной версии.
 *
 *
 * НЕ отвечает за:
 *
 * - текущую активную версию;
 * - поиск подходящего Skill;
 * - обучение;
 * - Planner;
 * - Earnings.
 *
 * =========================================================
 */


const EXPERIENCE_HISTORY_TABLE =
    "jessica_experience_skill_versions";


/*
 * =========================================================
 * NORMALIZE ID
 * =========================================================
 */


function normalizeSkillId(
    value
) {

    return String(
        value || ""
    ).trim();

}


/*
 * =========================================================
 * NORMALIZE VERSION
 * =========================================================
 */


function normalizeVersion(
    value
) {

    const version =
        Number(
            value
        );


    if (
        !Number.isInteger(version) ||
        version < 1
    ) {

        return null;

    }


    return version;

}


/*
 * =========================================================
 * SAVE VERSION
 * =========================================================
 *
 * Сохраняет снимок конкретной версии Skill.
 *
 * Уже существующая версия
 * не должна молча перезаписываться.
 *
 * =========================================================
 */


export async function saveExperienceVersion(
    experience
) {

    if (
        !experience ||
        typeof experience !== "object"
    ) {

        throw new Error(
            "Experience не указан"
        );

    }


    const skillId =
        normalizeSkillId(
            experience.id ||
            experience.skillId
        );


    const version =
        normalizeVersion(
            experience.version
        );


    if (!skillId) {

        throw new Error(
            "Skill ID не указан"
        );

    }


    if (!version) {

        throw new Error(
            "Некорректная версия Skill"
        );

    }


    const supabase =
        getSupabaseClient();


    const {
        data,
        error
    } =
        await supabase
            .from(
                EXPERIENCE_HISTORY_TABLE
            )
            .insert({

                skill_id:
                    skillId,

                version,

                payload: {
                    ...experience,
                    id:
                        skillId,
                    version
                }

            })
            .select(
                "skill_id, version, payload, created_at"
            )
            .single();


    if (error) {

        throw new Error(
            `Не удалось сохранить версию Experience: ${error.message}`
        );

    }


    return data;

}


/*
 * =========================================================
 * LOAD HISTORY
 * =========================================================
 */


export async function loadExperienceHistory(
    skillId
) {

    const id =
        normalizeSkillId(
            skillId
        );


    if (!id) {

        throw new Error(
            "Skill ID не указан"
        );

    }


    const supabase =
        getSupabaseClient();


    const {
        data,
        error
    } =
        await supabase
            .from(
                EXPERIENCE_HISTORY_TABLE
            )
            .select(
                "skill_id, version, payload, created_at"
            )
            .eq(
                "skill_id",
                id
            )
            .order(
                "version",
                {
                    ascending:
                        false
                }
            );


    if (error) {

        throw new Error(
            `Не удалось загрузить историю Experience: ${error.message}`
        );

    }


    return Array.isArray(data)
        ? data
        : [];

}


/*
 * =========================================================
 * LOAD VERSION
 * =========================================================
 */


export async function loadExperienceVersion(

    skillId,

    version

) {

    const id =
        normalizeSkillId(
            skillId
        );


    const normalizedVersion =
        normalizeVersion(
            version
        );


    if (!id) {

        throw new Error(
            "Skill ID не указан"
        );

    }


    if (!normalizedVersion) {

        throw new Error(
            "Некорректная версия Skill"
        );

    }


    const supabase =
        getSupabaseClient();


    const {
        data,
        error
    } =
        await supabase
            .from(
                EXPERIENCE_HISTORY_TABLE
            )
            .select(
                "skill_id, version, payload, created_at"
            )
            .eq(
                "skill_id",
                id
            )
            .eq(
                "version",
                normalizedVersion
            )
            .maybeSingle();


    if (error) {

        throw new Error(
            `Не удалось загрузить версию Experience: ${error.message}`
        );

    }


    return data || null;

          }
