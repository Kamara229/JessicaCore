import {
    getSupabaseClient
} from "../../storage/supabaseClient.js";


/*
 * =========================================================
 * JESSICA EXPERIENCE WRITER
 * =========================================================
 *
 * Отвечает только за атомарное сохранение
 * новой версии Skill в Supabase.
 *
 *
 * Использует PostgreSQL функцию:
 *
 * save_jessica_experience_skill()
 *
 *
 * Одна операция одновременно:
 *
 * 1. сохраняет версию в History;
 * 2. обновляет Current Skill.
 *
 *
 * Если один этап завершается ошибкой,
 * PostgreSQL откатывает всю операцию.
 *
 *
 * Этот модуль НЕ:
 *
 * - ищет Skills;
 * - загружает Skills;
 * - отключает Skills;
 * - выполняет Learning;
 * - вызывает Planner;
 * - содержит Earnings.
 *
 * =========================================================
 */


/*
 * =========================================================
 * RPC
 * =========================================================
 */


const SAVE_SKILL_RPC =
    "save_jessica_experience_skill";


/*
 * =========================================================
 * NORMALIZE SKILL ID
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
 * SAVE EXPERIENCE ATOMIC
 * =========================================================
 */


export async function saveExperienceAtomic(
    experience
) {


    /*
     * =====================================================
     * INPUT
     * =====================================================
     */


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



    /*
     * =====================================================
     * NORMALIZED SKILL
     * =====================================================
     */


    const skill = {

        ...experience,

        id:
            skillId,

        version,

        enabled:
            experience.enabled !== false

    };



    /*
     * =====================================================
     * SUPABASE
     * =====================================================
     */


    const supabase =
        getSupabaseClient();



    /*
     * =====================================================
     * ATOMIC DATABASE OPERATION
     * =====================================================
     */


    const {
        data,
        error
    } =
        await supabase.rpc(

            SAVE_SKILL_RPC,

            {

                p_skill_id:
                    skill.id,

                p_version:
                    skill.version,

                p_enabled:
                    skill.enabled,

                p_payload:
                    skill

            }

        );



    /*
     * =====================================================
     * ERROR
     * =====================================================
     */


    if (error) {

        throw new Error(
            `Не удалось атомарно сохранить Experience: ${error.message}`
        );

    }



    /*
     * =====================================================
     * RESULT
     * =====================================================
     */


    return {

        success:
            data?.success === true,

        id:
            data?.id ||
            skill.id,

        version:
            Number(
                data?.version ||
                skill.version
            ),

        enabled:
            data?.enabled !== false,

        experience:
            skill

    };


}
