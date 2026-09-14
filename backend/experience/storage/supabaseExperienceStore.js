import {
    getSupabaseClient
} from "../../storage/supabaseClient.js";


/*
 * =========================================================
 * JESSICA SUPABASE EXPERIENCE STORE
 * =========================================================
 *
 * Низкоуровневая работа Experience
 * с Supabase.
 *
 *
 * Отвечает только за:
 *
 * - чтение Skills;
 * - сохранение Skill;
 * - обновление Skill.
 *
 *
 * Не содержит:
 *
 * - поиска подходящего опыта;
 * - Learning;
 * - Planner;
 * - Earnings;
 * - бизнес-логики.
 *
 *
 * Таблица:
 *
 * jessica_experience_skills
 *
 * =========================================================
 */


const EXPERIENCE_TABLE =
    "jessica_experience_skills";


/*
 * =========================================================
 * NORMALIZE SKILL
 * =========================================================
 */


function normalizeSkill(
    value
) {

    if (
        !value ||
        typeof value !== "object"
    ) {

        return null;

    }


    const id =
        String(
            value.id ||
            value.skillId ||
            ""
        ).trim();


    if (!id) {

        return null;

    }


    return {

        ...value,

        id,

        enabled:
            value.enabled !== false,

        version:
            Number.isFinite(
                Number(
                    value.version
                )
            )
                ? Number(
                    value.version
                )
                : 1,

        confidence:
            Number.isFinite(
                Number(
                    value.confidence
                )
            )
                ? Number(
                    value.confidence
                )
                : 0

    };

}


/*
 * =========================================================
 * LOAD EXPERIENCES
 * =========================================================
 */


export async function loadExperiences() {

    const supabase =
        getSupabaseClient();


    const {
        data,
        error
    } =
        await supabase
            .from(
                EXPERIENCE_TABLE
            )
            .select(
                "id, payload, enabled, version"
            )
            .eq(
                "enabled",
                true
            );


    if (error) {

        throw new Error(
            `Не удалось загрузить Experience: ${error.message}`
        );

    }


    if (
        !Array.isArray(
            data
        )
    ) {

        return [];

    }


    return data
        .map(
            row => {


                const skill =
                    normalizeSkill(
                        {
                            ...(row.payload || {}),

                            id:
                                row.id,

                            enabled:
                                row.enabled,

                            version:
                                row.version
                        }
                    );


                return skill;

            }
        )
        .filter(
            Boolean
        );

}


/*
 * =========================================================
 * SAVE EXPERIENCE
 * =========================================================
 *
 * Используется как для создания,
 * так и для обновления Skill.
 *
 * =========================================================
 */


export async function saveExperience(
    experience
) {

    const skill =
        normalizeSkill(
            experience
        );


    if (!skill) {

        throw new Error(
            "Experience содержит некорректные данные"
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
                EXPERIENCE_TABLE
            )
            .upsert(
                {

                    id:
                        skill.id,

                    version:
                        skill.version,

                    enabled:
                        skill.enabled !== false,

                    payload:
                        skill,

                    updated_at:
                        new Date()
                            .toISOString()

                },
                {

                    onConflict:
                        "id"

                }
            )
            .select(
                "id, payload, enabled, version"
            )
            .single();


    if (error) {

        throw new Error(
            `Не удалось сохранить Experience: ${error.message}`
        );

    }


    return normalizeSkill(
        {

            ...(data?.payload || {}),

            id:
                data?.id,

            enabled:
                data?.enabled,

            version:
                data?.version

        }
    );

}


/*
 * =========================================================
 * DISABLE EXPERIENCE
 * =========================================================
 *
 * Skill не удаляем физически.
 *
 * Это важно для будущей истории обучения:
 *
 * плохой Skill можно отключить,
 * но сохранить для анализа или отката.
 *
 * =========================================================
 */


export async function disableExperience(
    skillId
) {

    const id =
        String(
            skillId || ""
        ).trim();


    if (!id) {

        throw new Error(
            "Skill ID не указан"
        );

    }


    const supabase =
        getSupabaseClient();


    const {
        error
    } =
        await supabase
            .from(
                EXPERIENCE_TABLE
            )
            .update(
                {

                    enabled:
                        false,

                    updated_at:
                        new Date()
                            .toISOString()

                }
            )
            .eq(
                "id",
                id
            );


    if (error) {

        throw new Error(
            `Не удалось отключить Experience: ${error.message}`
        );

    }


    return {

        success:
            true,

        id

    };

}
