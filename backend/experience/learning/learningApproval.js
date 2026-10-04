/*
 * =========================================================
 * JESSICA LEARNING APPROVAL v7
 * =========================================================
 *
 * Финальный исполнитель
 * Autonomous Learning Approval.
 *
 *
 * Flow:
 *
 * AUTO_APPROVE Proposal
 *        ↓
 * proposalId idempotency lookup
 *        ↓
 *
 * already saved?
 *      ├── YES → return existing
 *      │
 *      └── NO
 *           ↓
 *       Experience History
 *           ↓
 *       Version State
 *           ↓
 *       Build Experience
 *           ↓
 *       Atomic Save
 *           ↓
 *       Published Experience
 *
 *
 * НЕ:
 *
 * - запускает Reviewer;
 * - запускает Quality Gate;
 * - принимает Autonomy Decision;
 * - обновляет persistent Proposal status.
 *
 * =========================================================
 */


import {
    approveLearningProposal
} from "./learningProposal.js";


import {
    buildExperienceSkill,
    buildLearningSkillId
} from "./learningSkillBuilder.js";


import {
    getExperienceHistory,
    getExperienceByProposalId,
    saveExperienceSkill
} from "../storage/experienceStorage.js";


import {
    resolveExperienceVersionState
} from "../storage/experiencePersistence/experienceVersioning.js";





function normalizeText(
    value
) {

    return String(
        value || ""
    )
    .trim();

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


function buildFailure({

    stage,

    error,

    proposal = null

} = {}) {

    return {

        success:
            false,

        learned:
            false,

        existing:
            false,

        stage,

        error,

        proposal,

        experience:
            null

    };

}





/*
 * =========================================================
 * SKILL ID
 * =========================================================
 */


function resolveSkillId(
    proposal
) {

    const experience =

        isObject(
            proposal?.proposedExperience
        )

            ? proposal.proposedExperience

            : {};


    return (

        normalizeText(
            proposal?.targetSkill?.id
        )

        ||

        normalizeText(
            experience.id
        )

        ||

        normalizeText(
            experience.skillId
        )

        ||

        (
            normalizeText(
                experience.name
            )

                ? buildLearningSkillId(
                    experience.name
                )

                : null
        )

    );

}





/*
 * =========================================================
 * EXISTING RESULT
 * =========================================================
 */


function buildExistingResult({

    proposal,

    existing

}) {

    return {

        success:
            true,

        learned:
            true,

        existing:
            true,

        stage:
            "already-saved",

        proposal,

        experience:

            existing.payload

            ||

            null,

        skillId:

            existing.skillId

            ||

            existing.payload?.id

            ||

            proposal?.targetSkill?.id

            ||

            null,

        version:

            Number(
                existing.version
            ),

        previousVersion:

            existing.previousVersion

            ??

            existing.payload?.previousVersion

            ??

            null,

        action:
            proposal.action

    };

}





/*
 * =========================================================
 * MAIN
 * =========================================================
 */


export async function approveAndSaveLearningProposal({

    proposal,

    autonomy,

    confidence = null

} = {}) {


    /*
     * =====================================================
     * 1. INPUT
     * =====================================================
     */


    if(
        !isObject(
            proposal
        )
    ){

        return buildFailure({

            stage:
                "input",

            error:
                "Proposal отсутствует"

        });

    }


    if(
        !autonomy
        ||
        autonomy.action !== "AUTO_APPROVE"
    ){

        return buildFailure({

            stage:
                "autonomy",

            proposal,

            error:
                "Autonomy не разрешил обучение"

        });

    }


    if(
        !isObject(
            proposal.proposedExperience
        )
    ){

        return buildFailure({

            stage:
                "experience",

            proposal,

            error:
                "Нет proposedExperience"

        });

    }



    /*
     * =====================================================
     * 2. IDEMPOTENCY BEFORE VERSIONING
     * =====================================================
     */


    if(
        proposal.id
    ){

        try {


            const existing =

                await getExperienceByProposalId(
                    proposal.id
                );


            if(
                existing
            ){

                return buildExistingResult({

                    proposal,

                    existing

                });

            }


        }catch(error){


            return buildFailure({

                stage:
                    "idempotency",

                proposal,

                error:

                    error?.message

                    ||

                    "Experience idempotency lookup failed"

            });

        }

    }



    /*
     * =====================================================
     * 3. SKILL ID
     * =====================================================
     */


    const skillId =

        resolveSkillId(
            proposal
        );


    if(
        !skillId
    ){

        return buildFailure({

            stage:
                "skill",

            proposal,

            error:
                "Skill ID отсутствует"

        });

    }



    /*
     * =====================================================
     * 4. HISTORY
     * =====================================================
     */


    let history;


    try {


        history =

            await getExperienceHistory(
                skillId
            );


    }catch(error){


        return buildFailure({

            stage:
                "history",

            proposal,

            error:

                error?.message

                ||

                "Не удалось получить Experience History"

        });

    }



    /*
     * =====================================================
     * 5. VERSION STATE
     * =====================================================
     */


    const versionState =

        resolveExperienceVersionState(
            history
        );


    /*
     * Improvement без History
     * не должен создать v1.
     */


    if(
        proposal.action ===
        "SKILL_IMPROVEMENT"

        &&

        versionState.exists !== true
    ){

        return buildFailure({

            stage:
                "history",

            proposal,

            error:
                "SKILL_IMPROVEMENT не имеет существующей Experience History"

        });

    }



    /*
     * =====================================================
     * 6. BUILD
     * =====================================================
     */


    let experience;


    try {


        experience =

            buildExperienceSkill({

                proposedExperience:

                    proposal.proposedExperience,


                skillId,


                version:

                    versionState.nextVersion,


                previousVersion:

                    versionState.previousVersion,


                mode:

                    proposal.action ===
                    "SKILL_IMPROVEMENT"

                        ? "update"

                        : "create",


                confidence:

                    confidence

                    ??

                    proposal.confidence

                    ??

                    null,


                metadata: {

                    proposalId:

                        proposal.id

                        ||

                        null,


                    queueItemId:

                        proposal.queueItemId

                        ||

                        null,


                    traceId:

                        proposal.traceId

                        ||

                        null,


                    learnedFrom:

                        proposal.source

                        ||

                        "learning_pipeline",


                    candidateType:

                        proposal
                            ?.provenance
                            ?.candidateType

                        ||

                        proposal.action

                        ||

                        null,


                    improvementType:

                        proposal
                            ?.proposedExperience
                            ?.improvementType

                        ||

                        null,


                    baseVersion:

                        proposal
                            ?.proposedExperience
                            ?.baseVersion

                        ??

                        proposal
                            ?.targetSkill
                            ?.version

                        ??

                        null,


                    dynamicPattern:

                        proposal
                            ?.provenance
                            ?.dynamicPattern === true,


                    learning: {

                        action:

                            proposal.action,


                        autonomy:

                            autonomy.action,


                        reason:

                            autonomy.reason

                            ||

                            "",


                        metrics:

                            autonomy.metrics

                            ||

                            proposal
                                ?.proposedExperience
                                ?.learning

                            ||

                            null

                    }

                }

            });


    }catch(error){


        return buildFailure({

            stage:
                "builder",

            proposal,

            error:

                error?.message

                ||

                "Experience Skill Builder failed"

        });

    }



    /*
     * =====================================================
     * 7. SAVE
     * =====================================================
     */


    let saved;


    try {


        saved =

            await saveExperienceSkill(
                experience
            );


    }catch(error){


        return buildFailure({

            stage:
                "storage",

            proposal,

            error:

                error?.message

                ||

                "Experience Storage failed"

        });

    }


    if(
        saved?.success !== true
    ){

        return buildFailure({

            stage:
                "storage",

            proposal,

            error:

                saved?.error

                ||

                "Experience Storage не подтвердил сохранение"

        });

    }



    /*
     * =====================================================
     * 8. ACTUAL SAVED STATE
     * =====================================================
     *
     * Не используем versionState.nextVersion.
     *
     * RPC могла определить:
     *
     * existing=true
     *
     * и вернуть ранее сохранённую
     * фактическую версию.
     *
     * =====================================================
     */


    const actualSkillId =

        saved.skillId

        ||

        saved.experience?.id

        ||

        skillId;


    const actualVersion =

        Number(
            saved.version
        );


    const actualExperience =

        saved.experience

        ||

        experience;



    /*
     * =====================================================
     * 9. IN-MEMORY APPROVAL
     * =====================================================
     */


    let approvedProposal;


    try {


        approvedProposal =

            approveLearningProposal(
                proposal
            );


    }catch(error){


        /*
         * Experience уже сохранён.
         */


        return {

            success:
                true,

            learned:
                true,

            existing:

                saved.existing === true,

            stage:
                "saved",

            proposal,

            experience:
                actualExperience,

            skillId:
                actualSkillId,

            version:
                actualVersion,

            previousVersion:

                actualExperience?.previousVersion

                ??

                null,

            proposalStateUpdated:
                false,

            error:

                error?.message

                ||

                "Experience сохранён, но Proposal object не обновлён"

        };

    }



    /*
     * =====================================================
     * 10. SUCCESS
     * =====================================================
     */


    return {

        success:
            true,

        learned:
            true,

        existing:

            saved.existing === true,

        stage:

            saved.existing === true

                ? "already-saved"

                : "completed",

        proposal:
            approvedProposal,

        experience:
            actualExperience,

        skillId:
            actualSkillId,

        version:
            actualVersion,

        previousVersion:

            actualExperience?.previousVersion

            ??

            null,

        previousVersions:

            history.length,

        action:
            proposal.action

    };

}
