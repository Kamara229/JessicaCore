import {
    getSupabaseClient
} from "../../storage/supabaseClient.js";


/*
 * =========================================================
 * JESSICA EXPERIENCE WRITER v6
 * =========================================================
 *
 * Low-level adapter
 * Atomic Experience + Evidence Persistence.
 *
 *
 * Experience Skill
 *        +
 * Embedded Evidence Window
 *        ↓
 * Writer
 *        ↓
 * save_jessica_experience_skill_v3
 *        ↓
 * PostgreSQL Transaction
 *        │
 *        ├── Evidence Store
 *        │
 *        └── save_jessica_experience_skill_v2
 *              │
 *              ├── Current Skill
 *              └── Immutable History
 *
 *
 * ВАЖНО:
 *
 * На этапе Evidence Store v1
 * payload Skill пока НЕ меняется.
 *
 * examples сохраняются как раньше.
 *
 * Evidence Store работает параллельно
 * и позволяет безопасно проверить
 * новую архитектуру до ограничения
 * размера examples.
 *
 * =========================================================
 */


const SAVE_SKILL_RPC =
    "save_jessica_experience_skill_v3";





function normalizeText(
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





function normalizeBoolean(
    value
) {

    return value !== false;

}





function normalizePreviousVersion(
    value
) {

    if(
        value === null
        ||
        value === undefined
    ){

        return null;

    }


    return normalizeVersion(
        value
    );

}





function validateVersionChain(
    experience
) {

    const version =

        normalizeVersion(
            experience.version
        );


    const previous =

        normalizePreviousVersion(
            experience.previousVersion
        );


    if(
        version === 1
    ){

        return previous === null;

    }


    if(
        previous === null
    ){

        return false;

    }


    return (
        version === previous + 1
    );

}





/*
 * =========================================================
 * EVIDENCE
 * =========================================================
 */


function normalizeEvidence(
    value
) {

    if(
        !Array.isArray(
            value
        )
    ){

        return [];

    }


    return value

        .filter(
            item =>
                item
                &&
                typeof item === "object"
                &&
                !Array.isArray(item)
        )

        .map(
            item => ({
                ...item
            })
        );

}





function buildEvidence(
    experience
) {

    /*
     * Сейчас examples являются
     * единственным каноническим Evidence
     * внутри Experience Skill.
     *
     * RPC дедуплицирует их по
     * skill_id + evidence_key.
     *
     * Поэтому безопасно отправлять
     * текущее окно целиком.
     *
     * После перехода на bounded examples
     * сюда будут попадать только
     * последние/репрезентативные examples.
     */

    return normalizeEvidence(
        experience.examples
    );

}





/*
 * =========================================================
 * METADATA
 * =========================================================
 */


function buildMetadata(
    experience
) {

    return {

        ...(experience.metadata || {}),


        previousVersion:

            normalizePreviousVersion(
                experience.previousVersion
            ),


        mode:

            experience.mode

            ||

            "create",


        learningMode:

            experience.metadata?.learningMode

            ||

            "autonomous",


        createdBy:

            experience.metadata?.createdBy

            ||

            "jessica-learning",


        storage:

            "experience-writer",


        storageVersion:

            "v6",


        evidenceStorage:

            "jessica_experience_evidence",


        storedAt:

            new Date()
                .toISOString()

    };

}





/*
 * =========================================================
 * SAVE
 * =========================================================
 */


export async function saveExperienceAtomic(
    experience
) {

    if(
        !experience
        ||
        typeof experience !== "object"
        ||
        Array.isArray(experience)
    ){

        throw new Error(
            "Experience отсутствует"
        );

    }


    const skillId =

        normalizeText(

            experience.id

            ||

            experience.skillId

        );


    if(
        !skillId
    ){

        throw new Error(
            "Skill ID отсутствует"
        );

    }


    const version =

        normalizeVersion(
            experience.version
        );


    if(
        !version
    ){

        throw new Error(
            "Версия Skill некорректна"
        );

    }


    if(
        !validateVersionChain(
            experience
        )
    ){

        throw new Error(
            "Нарушена цепочка версий Experience"
        );

    }


    const previousVersion =

        normalizePreviousVersion(
            experience.previousVersion
        );


    /*
     * =====================================================
     * PAYLOAD
     * =====================================================
     */


    const payload = {

        ...experience,


        id:
            skillId,


        version,


        previousVersion,


        enabled:

            normalizeBoolean(
                experience.enabled
            )

    };


    /*
     * =====================================================
     * EVIDENCE
     * =====================================================
     */


    const evidence =

        buildEvidence(
            payload
        );


    /*
     * =====================================================
     * METADATA
     * =====================================================
     */


    const metadata =

        buildMetadata(
            payload
        );


    /*
     * =====================================================
     * RPC
     * =====================================================
     */


    const {
        data,
        error
    } =

        await getSupabaseClient()

            .rpc(

                SAVE_SKILL_RPC,

                {

                    p_skill_id:
                        skillId,

                    p_version:
                        version,

                    p_previous_version:
                        previousVersion,

                    p_mode:

                        metadata.mode

                        ||

                        "create",

                    p_enabled:
                        payload.enabled,

                    p_payload:
                        payload,

                    p_metadata:
                        metadata,

                    p_evidence:
                        evidence

                }

            );


    if(
        error
    ){

        throw new Error(

            "Experience atomic save failed: "

            +

            error.message

        );

    }


    const rpcResult =

        Array.isArray(data)

            ? data[0]

            : data;


    const success =

        rpcResult?.success === true

        ||

        rpcResult?.success === "true";


    if(
        !success
    ){

        return {

            success:
                false,

            error:

                rpcResult?.error

                ||

                "RPC не подтвердил сохранение"

        };

    }


    const storedVersion =

        normalizeVersion(
            rpcResult?.version
        )

        ||

        version;


    /*
     * =====================================================
     * RESULT
     * =====================================================
     */


    return {

        success:
            true,


        existing:

            rpcResult?.existing === true

            ||

            rpcResult?.existing === "true",


        id:

            normalizeText(
                rpcResult?.id
            )

            ||

            skillId,


        skillId:

            normalizeText(
                rpcResult?.skillId
            )

            ||

            normalizeText(
                rpcResult?.id
            )

            ||

            skillId,


        version:

            storedVersion,


        proposalId:

            normalizeText(
                rpcResult?.proposalId
            )

            ||

            normalizeText(
                metadata.proposalId
            )

            ||

            null,


        enabled:

            rpcResult?.enabled !== false,


        status:

            normalizeText(
                rpcResult?.status
            )

            ||

            null,


        auditId:

            rpcResult?.auditId

            ??

            rpcResult?.audit_id

            ??

            null,


        /*
         * Evidence diagnostics.
         */


        evidenceReceived:

            Number(
                rpcResult?.evidenceReceived
                ??
                0
            ),


        evidenceInserted:

            Number(
                rpcResult?.evidenceInserted
                ??
                0
            ),


        evidenceStorage:

            normalizeText(
                rpcResult?.evidenceStorage
            )

            ||

            "jessica_experience_evidence",


        writerVersion:

            normalizeText(
                rpcResult?.writerVersion
            )

            ||

            "v3",


        experience:

            (
                rpcResult?.experience
                &&
                typeof rpcResult.experience === "object"
            )

                ? rpcResult.experience

                : payload

    };

}
