/*
 * =========================================================
 * JESSICA APPROVAL CANDIDATE MEMORY v1
 * =========================================================
 *
 * Синхронизирует финальное решение
 * Learning Approval
 * с Candidate Memory.
 *
 *
 * Proposal Decision
 *        ↓
 *
 * APPROVED
 *        ↓
 * Candidate → PROMOTED
 *
 *
 * REJECTED
 *        ↓
 * Candidate → REJECTED
 *
 *
 * KEEP_CANDIDATE
 *        ↓
 * Candidate остаётся ACTIVE
 *
 *
 * FAILED
 *        ↓
 * Candidate остаётся ACTIVE
 *
 *
 * ВАЖНО:
 *
 * Техническая ошибка Pipeline
 * не должна уничтожать накопленный
 * Candidate Evidence.
 *
 *
 * НЕ:
 *
 * - принимает Learning Decision;
 * - создаёт Candidate;
 * - объединяет Candidate;
 * - сохраняет Experience Skill.
 *
 * =========================================================
 */


import {
    setLearningCandidateMemoryStatus
} from "../learningCandidateMemory.js";


/*
 * =========================================================
 * CANDIDATE MEMORY ID
 * =========================================================
 */


function resolveCandidateMemoryId(
    proposal
) {

    return (

        proposal
            ?.candidateMemory
            ?.id

        ||

        proposal
            ?.analysis
            ?.candidateMemory
            ?.id

        ||

        proposal
            ?.provenance
            ?.candidateMemoryId

        ||

        null

    );

}


/*
 * =========================================================
 * NO-OP
 * =========================================================
 */


function buildSkippedResult(
    reason
) {

    return {

        success:
            true,

        updated:
            false,

        skipped:
            true,

        reason

    };

}


/*
 * =========================================================
 * PROMOTE
 * =========================================================
 */


export async function promoteCandidateMemory({

    proposal,

    skillId = null

} = {}) {


    const id =

        resolveCandidateMemoryId(
            proposal
        );


    /*
     * SKILL_IMPROVEMENT пока
     * не использует Candidate Memory.
     *
     * Поэтому отсутствие ID —
     * нормальный сценарий.
     */


    if(
        !id
    ){

        return buildSkippedResult(
            "Candidate Memory отсутствует"
        );

    }


    try {


        const result =

            await setLearningCandidateMemoryStatus({

                id,

                status:
                    "PROMOTED",

                proposalId:

                    proposal?.id

                    ||

                    null,

                skillId:

                    skillId

                    ||

                    proposal
                        ?.targetSkill
                        ?.id

                    ||

                    proposal
                        ?.proposedExperience
                        ?.id

                    ||

                    null

            });


        return {

            success:
                result?.success === true,

            updated:
                result?.success === true,

            skipped:
                false,

            status:
                "PROMOTED",

            candidateMemoryId:
                id,

            candidate:

                result?.candidate

                ||

                null,

            error:

                result?.success === true

                    ? null

                    : (
                        result?.error

                        ||

                        "Candidate Memory не переведён в PROMOTED"
                    )

        };


    }catch(error){


        return {

            success:
                false,

            updated:
                false,

            skipped:
                false,

            status:
                "PROMOTED",

            candidateMemoryId:
                id,

            error:

                error?.message

                ||

                "Candidate Memory promotion failed"

        };

    }

}


/*
 * =========================================================
 * REJECT
 * =========================================================
 */


export async function rejectCandidateMemory({

    proposal,

    reason = ""

} = {}) {


    const id =

        resolveCandidateMemoryId(
            proposal
        );


    if(
        !id
    ){

        return buildSkippedResult(
            "Candidate Memory отсутствует"
        );

    }


    try {


        const result =

            await setLearningCandidateMemoryStatus({

                id,

                status:
                    "REJECTED",

                proposalId:

                    proposal?.id

                    ||

                    null,

                skillId:

                    proposal
                        ?.targetSkill
                        ?.id

                    ||

                    null

            });


        return {

            success:
                result?.success === true,

            updated:
                result?.success === true,

            skipped:
                false,

            status:
                "REJECTED",

            candidateMemoryId:
                id,

            reason,

            candidate:

                result?.candidate

                ||

                null,

            error:

                result?.success === true

                    ? null

                    : (
                        result?.error

                        ||

                        "Candidate Memory не переведён в REJECTED"
                    )

        };


    }catch(error){


        return {

            success:
                false,

            updated:
                false,

            skipped:
                false,

            status:
                "REJECTED",

            candidateMemoryId:
                id,

            reason,

            error:

                error?.message

                ||

                "Candidate Memory rejection failed"

        };

    }

}


/*
 * =========================================================
 * KEEP ACTIVE
 * =========================================================
 *
 * Физически ничего в БД менять
 * не требуется.
 *
 * Candidate уже имеет status ACTIVE.
 *
 * =========================================================
 */


export function keepCandidateMemoryActive(
    proposal
) {

    const id =

        resolveCandidateMemoryId(
            proposal
        );


    if(
        !id
    ){

        return buildSkippedResult(
            "Candidate Memory отсутствует"
        );

    }


    return {

        success:
            true,

        updated:
            false,

        skipped:
            false,

        status:
            "ACTIVE",

        candidateMemoryId:
            id,

        reason:
            "Candidate Memory остаётся ACTIVE"

    };

}
