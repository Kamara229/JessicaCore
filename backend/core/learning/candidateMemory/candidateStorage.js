/*
 * =========================================================
 * JESSICA CANDIDATE MEMORY STORAGE v1
 * =========================================================
 *
 * Persistent storage
 * Learning Candidates.
 *
 *
 * Table:
 *
 * learning_candidates
 *
 *
 * НЕ:
 *
 * - объединяет Candidates;
 * - рассчитывает similarity;
 * - принимает AUTO_APPROVE.
 *
 * =========================================================
 */


import {
    getSupabaseClient
} from "../../../storage/supabaseClient.js";


const TABLE_NAME =
    "learning_candidates";


const ACTIVE_STATUS =
    "ACTIVE";


function getClient()
{

    return getSupabaseClient();

}


function isObject(
    value
) {

    return (

        value &&
        typeof value === "object" &&
        !Array.isArray(value)

    );

}


function normalizeRow(
    row
) {

    if(
        !isObject(
            row
        )
    ){

        return null;

    }


    return {

        id:
            row.id || null,

        candidateKey:
            row.candidate_key || null,

        action:
            row.action || null,

        skillId:
            row.skill_id || null,

        category:
            row.category || "general",

        status:
            row.status || null,

        confidence:
            Number(row.confidence || 0),

        occurrences:
            Number(row.occurrences || 0),

        candidate:

            isObject(
                row.candidate_json
            )

                ? row.candidate_json

                : {},

        metadata:

            isObject(
                row.metadata
            )

                ? row.metadata

                : {},

        createdAt:
            row.created_at || null,

        updatedAt:
            row.updated_at || null,

        lastSeenAt:
            row.last_seen_at || null

    };

}


/*
 * =========================================================
 * FIND EXACT
 * =========================================================
 */


export async function findActiveCandidateByKey(
    candidateKey
) {

    if(
        !candidateKey
    ){

        return null;

    }


    const {
        data,
        error
    } =

        await getClient()

            .from(
                TABLE_NAME
            )

            .select("*")

            .eq(
                "candidate_key",
                candidateKey
            )

            .eq(
                "status",
                ACTIVE_STATUS
            )

            .maybeSingle();


    if(
        error
    ){

        throw new Error(
            error.message
        );

    }


    return normalizeRow(
        data
    );

}


/*
 * =========================================================
 * LIST ACTIVE BY CATEGORY
 * =========================================================
 */


export async function listActiveCandidatesByCategory(
    category,
    limit = 50
) {

    const {
        data,
        error
    } =

        await getClient()

            .from(
                TABLE_NAME
            )

            .select("*")

            .eq(
                "status",
                ACTIVE_STATUS
            )

            .eq(
                "category",
                category || "general"
            )

            .order(
                "updated_at",
                {
                    ascending:
                        false
                }
            )

            .limit(
                limit
            );


    if(
        error
    ){

        throw new Error(
            error.message
        );

    }


    return (

        Array.isArray(data)

            ? data

            : []

    )
    .map(
        normalizeRow
    )
    .filter(
        Boolean
    );

}


/*
 * =========================================================
 * CREATE
 * =========================================================
 */


export async function createCandidateMemory({

    id,

    candidateKey,

    action,

    skillId,

    category,

    candidate,

    metadata = {}

}) {

    const now =

        new Date()
            .toISOString();


    const payload = {

        id,

        candidate_key:
            candidateKey,

        action,

        skill_id:
            skillId || null,

        category:
            category || "general",

        status:
            ACTIVE_STATUS,

        confidence:
            Number(candidate?.confidence || 0),

        occurrences:
            Number(candidate?.occurrences || 1),

        candidate_json:
            candidate,

        metadata,

        created_at:
            now,

        updated_at:
            now,

        last_seen_at:
            now

    };


    const {
        data,
        error
    } =

        await getClient()

            .from(
                TABLE_NAME
            )

            .insert(
                payload
            )

            .select()
            .single();


    if(
        error
    ){

        throw new Error(
            error.message
        );

    }


    return normalizeRow(
        data
    );

}


/*
 * =========================================================
 * UPDATE CANDIDATE
 * =========================================================
 */


export async function updateCandidateMemory({

    id,

    candidate,

    skillId = null,

    metadata = null

}) {

    if(
        !id
    ){

        throw new Error(
            "Candidate Memory ID отсутствует"
        );

    }


    const payload = {

        candidate_json:
            candidate,

        confidence:
            Number(candidate?.confidence || 0),

        occurrences:
            Number(candidate?.occurrences || 1),

        updated_at:

            new Date()
                .toISOString(),

        last_seen_at:

            new Date()
                .toISOString()

    };


    if(
        skillId
    ){

        payload.skill_id =
            skillId;

    }


    if(
        metadata
    ){

        payload.metadata =
            metadata;

    }


    const {
        data,
        error
    } =

        await getClient()

            .from(
                TABLE_NAME
            )

            .update(
                payload
            )

            .eq(
                "id",
                id
            )

            .select()
            .single();


    if(
        error
    ){

        throw new Error(
            error.message
        );

    }


    return normalizeRow(
        data
    );

}


/*
 * =========================================================
 * UPDATE STATUS
 * =========================================================
 */


export async function updateCandidateMemoryStatus({

    id,

    status,

    metadata = {}

}) {

    if(
        !id
    ){

        return {

            success:
                false,

            error:
                "Candidate Memory ID отсутствует"

        };

    }


    try {


        const {
            data:current,
            error:readError
        } =

            await getClient()

                .from(
                    TABLE_NAME
                )

                .select(
                    "metadata"
                )

                .eq(
                    "id",
                    id
                )

                .single();


        if(
            readError
        ){

            return {

                success:
                    false,

                error:
                    readError.message

            };

        }


        const previousMetadata =

            isObject(
                current?.metadata
            )

                ? current.metadata

                : {};


        const {
            data,
            error
        } =

            await getClient()

                .from(
                    TABLE_NAME
                )

                .update({

                    status,

                    metadata: {

                        ...previousMetadata,

                        ...metadata

                    },

                    updated_at:

                        new Date()
                            .toISOString()

                })

                .eq(
                    "id",
                    id
                )

                .select()
                .single();


        if(
            error
        ){

            return {

                success:
                    false,

                error:
                    error.message

            };

        }


        return {

            success:
                true,

            candidate:
                normalizeRow(data)

        };


    }catch(error){


        return {

            success:
                false,

            error:

                error?.message

                ||

                "Candidate Memory status update failed"

        };

    }

      }
