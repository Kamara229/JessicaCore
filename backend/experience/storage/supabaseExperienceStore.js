import {
    getSupabaseClient
} from "../../storage/supabaseClient.js";


/*
 * =========================================================
 * JESSICA SUPABASE EXPERIENCE STORE v4
 * =========================================================
 *
 * Активная память Experience.
 *
 * Отвечает:
 *
 * - загрузка опубликованных и включённых Skills;
 * - получение активного Skill;
 * - отключение Skill.
 *
 * НЕ:
 *
 * - обучает;
 * - создаёт версии;
 * - сохраняет Skill.
 *
 * =========================================================
 */


const EXPERIENCE_TABLE =
    "jessica_experience_skills";


const ACTIVE_STATUS =
    "published";


function normalizeArray(value) {

    return Array.isArray(value)
        ? value.filter(Boolean)
        : [];

}


function normalizeSkill(value) {

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

        version:
            Number(
                value.version || 1
            ),

        status:
            String(
                value.status || ""
            )
            .trim()
            .toLowerCase(),

        enabled:
            value.enabled === true,

        confidence:
            Number(
                value.confidence || 0
            ),

        category:
            value.category ||
            "general",

        workflow:
            normalizeArray(
                value.workflow
            ),

        keywords:
            normalizeArray(
                value.keywords
            ),

        tags:
            normalizeArray(
                value.tags
            ),

        triggerPatterns:
            normalizeArray(
                value.triggerPatterns
            ),

        validationRules:
            normalizeArray(
                value.validationRules
            ),

        constraints:
            normalizeArray(
                value.constraints
            ),

        strategy:
            normalizeArray(
                value.strategy
            ),

        sourcePriority:
            normalizeArray(
                value.sourcePriority
            ),

        successfulPatterns:
            normalizeArray(
                value.successfulPatterns
            ),

        failurePatterns:
            normalizeArray(
                value.failurePatterns
            ),

        avoidPatterns:
            normalizeArray(
                value.avoidPatterns
            ),

        statistics: {

            successfulRuns:
                Number(
                    value.statistics
                        ?.successfulRuns
                    ||
                    0
                ),

            failedRuns:
                Number(
                    value.statistics
                        ?.failedRuns
                    ||
                    0
                ),

            lastUsedAt:
                value.statistics
                    ?.lastUsedAt
                ||
                null,

            lastResult:
                value.statistics
                    ?.lastResult
                ||
                null
        },

        metadata: {
            ...(value.metadata || {})
        }
    };

}


/*
 * =========================================================
 * LOAD ACTIVE EXPERIENCE
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

            .select(`
                id,
                payload,
                enabled,
                version,
                status
            `)

            .eq(
                "enabled",
                true
            )

            .eq(
                "status",
                ACTIVE_STATUS
            )

            .order(
                "version",
                {
                    ascending: false
                }
            );


    if (error) {

        throw new Error(
            `Experience load error: ${error.message}`
        );

    }


    if (!Array.isArray(data)) {
        return [];
    }


    const latest =
        new Map();


    for (const row of data) {

        const skill =
            normalizeSkill({

                ...(row.payload || {}),

                id:
                    row.id,

                version:
                    row.version,

                enabled:
                    row.enabled,

                status:
                    row.status
            });


        if (!skill) {
            continue;
        }


        /*
         * Defense in depth.
         *
         * Даже если запрос к БД когда-нибудь изменится,
         * Store не должен отдавать неактивный Skill.
         */
        if (
            skill.enabled !== true ||
            skill.status !== ACTIVE_STATUS
        ) {
            continue;
        }


        const current =
            latest.get(
                skill.id
            );


        if (
            !current ||
            skill.version > current.version
        ) {

            latest.set(
                skill.id,
                skill
            );

        }
    }


    return Array.from(
        latest.values()
    );

}


/*
 * =========================================================
 * LOAD SINGLE ACTIVE SKILL
 * =========================================================
 */


export async function loadExperienceSkill(
    skillId
) {

    const id =
        String(
            skillId || ""
        ).trim();


    if (!id) {
        return null;
    }


    const skills =
        await loadExperiences();


    return (
        skills.find(
            skill =>
                skill.id === id
        )
        ||
        null
    );

}


/*
 * =========================================================
 * DISABLE
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
            "Skill ID отсутствует"
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

                enabled: false,

                status: "disabled",

                updated_at:
                    new Date()
                        .toISOString()

            })

            .eq(
                "id",
                id
            )

            .select(
                "id, status, enabled"
            );


    if (error) {

        throw new Error(
            `Disable Experience error: ${error.message}`
        );

    }


    return {

        success:
            Array.isArray(data)
            &&
            data.length > 0,

        id,

        status:
            data?.[0]?.status
            ||
            null,

        enabled:
            data?.[0]?.enabled
            ??
            null
    };

}
