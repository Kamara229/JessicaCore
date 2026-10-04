import {
    getSupabaseClient
} from "../../storage/supabaseClient.js";


/*
 * =========================================================
 * JESSICA EXPERIENCE HISTORY STORE v4
 * =========================================================
 *
 * Immutable Experience History.
 *
 * =========================================================
 */


const EXPERIENCE_HISTORY_TABLE =
    "jessica_experience_skill_versions";





function normalizeSkillId(
    value
) {

    return String(
        value || ""
    )
    .trim();

}


function normalizeVersion(
    value
) {

    const version =
        Number(value);


    if(
        !Number.isInteger(version)
        ||
        version < 1
    ){

        return null;

    }


    return version;

}


function normalizeHistoryItem(
    row
) {

    if(
        !row ||
        typeof row !== "object"
    ){

        return null;

    }


    return {

        id:

            row.id

            ??

            null,

        skillId:

            row.skill_id

            ||

            row.payload?.id

            ||

            null,

        version:

            Number(
                row.version
            ),

        previousVersion:

            row.previous_version

            ??

            row.payload?.previousVersion

            ??

            null,

        mode:

            row.mode

            ||

            row.payload?.mode

            ||

            "create",

        payload:

            row.payload

            ||

            {},

        metadata:

            row.metadata

            ||

            row.payload?.metadata

            ||

            {},

        createdAt:

            row.created_at

            ||

            null

    };

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


    if(
        !id
    ){

        throw new Error(
            "Skill ID отсутствует"
        );

    }


    const {
        data,
        error
    } =

        await getSupabaseClient()

            .from(
                EXPERIENCE_HISTORY_TABLE
            )

            .select("*")

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


    if(
        error
    ){

        throw new Error(

            `History load error: ${error.message}`

        );

    }


    return (

        Array.isArray(data)

            ? data

            : []

    )
    .map(
        normalizeHistoryItem
    )
    .filter(
        Boolean
    );

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


    if(
        !id ||
        !normalizedVersion
    ){

        throw new Error(
            "Некорректная версия"
        );

    }


    const {
        data,
        error
    } =

        await getSupabaseClient()

            .from(
                EXPERIENCE_HISTORY_TABLE
            )

            .select("*")

            .eq(
                "skill_id",
                id
            )

            .eq(
                "version",
                normalizedVersion
            )

            .maybeSingle();


    if(
        error
    ){

        throw new Error(

            `Version load error: ${error.message}`

        );

    }


    return data

        ? normalizeHistoryItem(data)

        : null;

}


/*
 * =========================================================
 * LOAD BY PROPOSAL ID
 * =========================================================
 *
 * JSONB containment:
 *
 * payload @> {
 *   metadata: {
 *      proposalId: "..."
 *   }
 * }
 *
 * =========================================================
 */


export async function loadExperienceVersionByProposalId(
    proposalId
) {

    const id =

        normalizeSkillId(
            proposalId
        );


    if(
        !id
    ){

        return null;

    }


    const {
        data,
        error
    } =

        await getSupabaseClient()

            .from(
                EXPERIENCE_HISTORY_TABLE
            )

            .select("*")

            .contains(

                "payload",

                {

                    metadata: {

                        proposalId:
                            id

                    }

                }

            )

            .limit(1)

            .maybeSingle();


    if(
        error
    ){

        throw new Error(

            `Proposal Experience lookup error: ${error.message}`

        );

    }


    return data

        ? normalizeHistoryItem(data)

        : null;

}


/*
 * =========================================================
 * LOAD LATEST
 * =========================================================
 */


export async function loadLatestExperienceVersion(
    skillId
) {

    const history =

        await loadExperienceHistory(
            skillId
        );


    return history.length > 0

        ? history[0]

        : null;

}
