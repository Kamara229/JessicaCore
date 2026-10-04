/*
 * =========================================================
 * JESSICA EXPERIENCE PERSISTENCE WRITER v1
 * =========================================================
 *
 * Канонический write-side.
 *
 *
 * Experience
 *      ↓
 * normalize
 *      ↓
 * proposalId pre-check
 *      ↓
 * atomic PostgreSQL save
 *
 *
 * PostgreSQL остаётся финальным
 * источником гарантии idempotency.
 *
 * =========================================================
 */


import {
    saveExperienceAtomic
} from "../supabaseExperienceWriter.js";


import {
    normalizeExperienceSkill
} from "./experienceNormalizer.js";


import {
    findExperienceByProposalId
} from "./experienceIdempotency.js";





function buildValidationError(
    error
) {

    return {

        success:
            false,

        existing:
            false,

        error

    };

}





export async function persistExperienceSkill(
    experience
) {

    const normalized =

        normalizeExperienceSkill(
            experience
        );


    if(
        !normalized
    ){

        return buildValidationError(
            "Experience отсутствует"
        );

    }


    if(
        !normalized.id
    ){

        return buildValidationError(
            "Skill ID отсутствует"
        );

    }


    if(
        !normalized.name
    ){

        return buildValidationError(
            "Skill name отсутствует"
        );

    }


    if(
        !normalized.version
    ){

        return buildValidationError(
            "Skill version отсутствует"
        );

    }


    const proposalId =

        String(

            normalized
                ?.metadata
                ?.proposalId

            ||

            ""

        )
        .trim();


    /*
     * =====================================================
     * FAST IDEMPOTENCY CHECK
     * =====================================================
     *
     * Это оптимизация.
     *
     * Финальная гарантия всё равно
     * находится внутри PostgreSQL RPC.
     *
     * =====================================================
     */


    if(
        proposalId
    ){

        const existing =

            await findExperienceByProposalId(
                proposalId
            );


        if(
            existing
        ){

            return {

                success:
                    true,

                existing:
                    true,

                skillId:

                    existing.skillId

                    ||

                    existing.payload?.id

                    ||

                    normalized.id,

                version:

                    Number(
                        existing.version
                    ),

                experience:

                    existing.payload

                    ||

                    normalized,

                history:

                    existing,

                storage:

                    "idempotency-precheck"

            };

        }

    }


    /*
     * =====================================================
     * STORAGE META
     * =====================================================
     */


    normalized.metadata = {

        ...normalized.metadata,

        storage:

            "experience-storage",

        storageVersion:

            "v5",

        storedAt:

            new Date()
                .toISOString()

    };


    /*
     * =====================================================
     * ATOMIC SAVE
     * =====================================================
     */


    const result =

        await saveExperienceAtomic(
            normalized
        );


    if(
        !result?.success
    ){

        return {

            success:
                false,

            existing:
                false,

            error:

                result?.error

                ||

                "Ошибка сохранения Experience"

        };

    }


    /*
     * Если RPC определил повторный
     * proposalId, возвращаем именно
     * фактически сохранённую версию,
     * а не version из нового запроса.
     */


    return {

        success:
            true,

        existing:

            result.existing === true,

        skillId:

            result.skillId

            ||

            result.id

            ||

            normalized.id,

        version:

            Number(
                result.version
            ),

        experience:

            result.experience

            ||

            normalized,

        status:

            result.status

            ||

            null,

        storage:

            result

    };

}
