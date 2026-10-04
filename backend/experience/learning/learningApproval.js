/*
 * =========================================================
 * JESSICA LEARNING APPROVAL v6
 * =========================================================
 *
 * Финальный исполнитель создания
 * и сохранения Experience Skill.
 *
 *
 * Flow:
 *
 * Approved Learning Proposal
 *        ↓
 * Resolve Skill ID
 *        ↓
 * Experience History
 *        ↓
 * Resolve Version
 *        ↓
 * Build Experience Skill
 *        ↓
 * Experience Storage
 *        ↓
 * In-memory Proposal Approval
 *
 *
 * НЕ:
 *
 * - принимает Learning Decision;
 * - запускает Reviewer;
 * - запускает Quality Gate;
 * - запускает Autonomy Policy;
 * - обновляет Proposal в Supabase;
 * - вызывает AI.
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
    saveExperienceSkill
} from "../storage/experienceStorage.js";


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
 * HISTORY
 * =========================================================
 */


function normalizeHistory(
    value
) {

    return Array.isArray(
        value
    )

        ? value

        : [];

}


function resolveHistoryVersion(
    item
) {

    const version =

        Number(

            item?.version

            ??

            item?.payload?.version

        );


    return (

        Number.isInteger(version) &&
        version > 0

    )

        ? version

        : null;

}


function resolveVersionState(
    history
) {

    const versions =

        normalizeHistory(
            history
        )

        .map(
            resolveHistoryVersion
        )

        .filter(
            Number.isInteger
        );


    if(
        versions.length === 0
    ){

        return {

            version:
                1,

            previousVersion:
                null,

            historyCount:
                0

        };

    }


    const latestVersion =

        Math.max(
            ...versions
        );


    return {

        version:

            latestVersion + 1,

        previousVersion:

            latestVersion,

        historyCount:

            versions.length

    };

}


/*
 * =========================================================
 * VALIDATE HISTORY AGAINST ACTION
 * =========================================================
 */


function validateHistoryForAction({

    proposal,

    history

}) {


    const action =
        proposal?.action;


    const hasHistory =

        Array.isArray(history) &&
        history.length > 0;


    /*
     * Improvement без существующего Skill
     * не должен случайно создать v1.
     */


    if(
        action === "SKILL_IMPROVEMENT"
        &&
        !hasHistory
    ){

        return {

            valid:
                false,

            reason:
                "SKILL_IMPROVEMENT не имеет существующей Experience history"

        };

    }


    return {

        valid:
            true

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
        !autonomy ||
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
     * 2. SKILL ID
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
     * 3. HISTORY
     * =====================================================
     */


    let history;


    try {


        history =

            normalizeHistory(

                await getExperienceHistory(
                    skillId
                )

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


    const historyValidation =

        validateHistoryForAction({

            proposal,

            history

        });


    if(
        !historyValidation.valid
    ){

        return buildFailure({

            stage:
                "history",

            proposal,

            error:
                historyValidation.reason

        });

    }


    const versionState =

        resolveVersionState(
            history
        );


    /*
     * =====================================================
     * 4. BUILD EXPERIENCE
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

                    versionState.version,

                previousVersion:

                    versionState.previousVersion,

                mode:

                    proposal.action ===
                    "SKILL_IMPROVEMENT"

                        ? "update"

                        : "create",

                /*
                 * Нет искусственного 0.7.
                 *
                 * Skill Builder сам умеет
                 * читать proposedExperience.learning.
                 */

                confidence:

                    confidence

                    ??

                    proposal.confidence

                    ??

                    null,

                metadata: {

                    proposalId:

                        proposal.id ||

                        null,

                    queueItemId:

                        proposal.queueItemId ||

                        null,

                    traceId:

                        proposal.traceId ||

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

                            autonomy.reason || "",

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
     * 5. SAVE EXPERIENCE
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
        !saved?.success
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
     * 6. IN-MEMORY PROPOSAL APPROVAL
     * =====================================================
     *
     * Persistent status обновит
     * Learning Approval Runner.
     *
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
         * Skill уже сохранён.
         *
         * Поэтому нельзя возвращать
         * learned=false.
         */


        return {

            success:
                true,

            learned:
                true,

            stage:
                "saved",

            proposal,

            experience,

            skillId,

            version:
                versionState.version,

            previousVersion:
                versionState.previousVersion,

            previousVersions:
                versionState.historyCount,

            proposalStateUpdated:
                false,

            error:

                error?.message

                ||

                "Skill сохранён, но Proposal object не обновлён"

        };

    }


    /*
     * =====================================================
     * 7. SUCCESS
     * =====================================================
     */


    return {

        success:
            true,

        learned:
            true,

        stage:
            "completed",

        proposal:
            approvedProposal,

        experience,

        skillId,

        version:
            versionState.version,

        previousVersion:
            versionState.previousVersion,

        previousVersions:
            versionState.historyCount,

        action:
            proposal.action

    };

}
