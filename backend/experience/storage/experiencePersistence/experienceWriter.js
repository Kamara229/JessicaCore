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
import {recoverExperienceSaveReceipt} from "./experienceAckRecovery.js";





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
    experience,{lookup=findExperienceByProposalId,save=saveExperienceAtomic}={}
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

        let existing;
        try {
            existing=await lookup(proposalId);
        }catch(error){
            // The preflight read did not establish that a prior commit is
            // absent. Never risk another version while storage is unavailable.
            return {success:false,existing:false,uncertain:true,
                error:"Experience preflight lookup unavailable: "+
                    (error?.message||"unknown")};
        }


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


    let result;
    let writeError=null;
    try {
        result=await save(normalized);
    }catch(error){
        writeError=error?.message||"Experience atomic save ACK lost";
    }
    if(!result?.success){
        return recoverExperienceSaveReceipt({
            proposalId,skillId:normalized.id,lookup,
            reason:writeError||result?.error||"Experience save unconfirmed"
        });
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
