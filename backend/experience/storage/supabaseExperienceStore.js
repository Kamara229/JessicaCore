import {
    getSupabaseClient
} from "../../storage/supabaseClient.js";


/*
 * =========================================================
 * JESSICA SUPABASE EXPERIENCE STORE
 * =========================================================
 *
 * Низкоуровневая работа
 * с текущими Skills в Supabase.
 *
 *
 * Отвечает только за:
 *
 * - чтение активных Skills;
 * - отключение текущего Skill.
 *
 *
 * Сохранение новых версий
 * здесь ЗАПРЕЩЕНО.
 *
 * Новые версии сохраняются только через:
 *
 * supabaseExperienceWriter.js
 *
 * который использует атомарную PostgreSQL-функцию:
 *
 * save_jessica_experience_skill()
 *
 *
 * Это гарантирует:
 *
 * History + Current
 *
 * обновляются одной транзакцией.
 *
 *
 * Этот модуль НЕ содержит:
 *
 * - сохранение новой версии Skill;
 * - историю версий;
 * - поиск подходящего опыта;
 * - Learning;
 * - Planner;
 * - Earnings;
 * - бизнес-логику.
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


    const rawVersion =
        Number(
            value.version
        );


    const version =
        Number.isInteger(
            rawVersion
        ) &&
        rawVersion >= 1
            ? rawVersion
            : 1;


    const rawConfidence =
        Number(
            value.confidence
        );


    const confidence =
        Number.isFinite(
            rawConfidence
        )
            ? rawConfidence
            : 0;


    return {

        ...value,

        id,

        version,

        enabled:
            value.enabled !== false,

        confidence

    };


}


/*
 * =========================================================
 * LOAD EXPERIENCES
 * =========================================================
 *
 * Загружает только активные
 * текущие Skills Jessica.
 *
 * История версий здесь
 * не загружается.
 *
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


                return normalizeSkill({

                    ...(row.payload || {}),

                    id:
                        row.id,

                    enabled:
                        row.enabled,

                    version:
                        row.version

                });


            }
        )
        .filter(
            Boolean
        );


}


/*
 * =========================================================
 * DISABLE EXPERIENCE
 * =========================================================
 *
 * Отключает текущий Skill,
 * но физически его не удаляет.
 *
 *
 * Это позволяет сохранить:
 *
 * - историю;
 * - версии;
 * - возможность анализа;
 * - возможность будущего восстановления.
 *
 *
 * Отключённый Skill больше
 * не будет возвращаться через
 * loadExperiences().
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
        data,
        error
    } =
        await supabase
            .from(
                EXPERIENCE_TABLE
            )
            .update({

                enabled:
                    false,

                updated_at:
                    new Date()
                        .toISOString()

            })
            .eq(
                "id",
                id
            )
            .select(
                "id"
            );


    if (error) {

        throw new Error(
            `Не удалось отключить Experience: ${error.message}`
        );

    }


    /*
     * Если Skill с таким ID
     * не существует, update не является
     * SQL-ошибкой.
     *
     * Поэтому отдельно проверяем,
     * была ли найдена запись.
     */


    if (
        !Array.isArray(
            data
        ) ||
        data.length === 0
    ) {

        return {

            success:
                false,

            id,

            reason:
                "Experience не найден"

        };

    }


    return {

        success:
            true,

        id

    };


}
