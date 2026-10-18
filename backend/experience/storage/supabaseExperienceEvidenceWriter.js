/*
 * =========================================================
 * JESSICA EXPERIENCE EVIDENCE WRITER v1
 * =========================================================
 *
 * Сохраняет новое подтверждение READY Skill
 * без создания новой версии Skill.
 *
 * Используется Skill Evolution Guard:
 *
 * READY + knowledge unchanged
 *      ↓
 * Evidence only
 *      ↓
 * no version churn
 * =========================================================
 */


import {
    getSupabaseClient
} from "../../storage/supabaseClient.js";


const APPEND_EVIDENCE_RPC =
    "append_jessica_experience_evidence_v1";


function normalizeText(
    value
) {

    return String(
        value || ""
    )
        .trim();

}


function normalizeEvidence(
    value
) {

    return Array.isArray(value)

        ? value.filter(
            item =>
                item
                &&
                typeof item === "object"
                &&
                !Array.isArray(item)
        )

        : [];

}


export async function appendExperienceEvidence({

    skillId,

    evidence

} = {}) {

    const id =

        normalizeText(
            skillId
        );


    if(!id){

        return {

            success:
                false,

            error:
                "Skill ID отсутствует"

        };

    }


    const items =

        normalizeEvidence(
            evidence
        );


    if(
        items.length === 0
    ){

        return {

            success:
                false,

            error:
                "Evidence отсутствует"

        };

    }


    const {
        data,
        error
    } =

        await getSupabaseClient()

            .rpc(

                APPEND_EVIDENCE_RPC,

                {

                    p_skill_id:
                        id,

                    p_evidence:
                        items

                }

            );


    if(error){

        return {

            success:
                false,

            error:
                error.message

        };

    }


    const result =

        Array.isArray(data)

            ? data[0]

            : data;


    return {

        success:
            result?.success === true,

        skillId:

            normalizeText(
                result?.skillId
            )

            ||

            id,

        evidenceReceived:

            Number(
                result?.evidenceReceived || 0
            ),

        evidenceInserted:

            Number(
                result?.evidenceInserted || 0
            ),

        writerVersion:

            normalizeText(
                result?.writerVersion
            )

            ||

            "evidence-v1",

        error:

            result?.success === true

                ? null

                : (
                    result?.error

                    ||

                    "RPC не подтвердил Evidence save"
                )

    };

}
