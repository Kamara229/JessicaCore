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


/*
 * =========================================================
 * JESSICA LEARNING APPROVAL
 * =========================================================
 *
 * Финальный этап подтверждённого обучения.
 *
 *
 * Рабочая цепочка:
 *
 * PENDING Learning Proposal
 *        ↓
 * пользователь подтверждает
 *        ↓
 * определяем Skill ID
 *        ↓
 * читаем историю Skill
 *        ↓
 * определяем следующую version
 *        ↓
 * buildExperienceSkill()
 *        ↓
 * saveExperienceSkill()
 *        ↓
 * атомарно:
 *
 * History vN
 * +
 * Current vN
 *
 *        ↓
 * Proposal → APPROVED
 *
 *
 * ВАЖНО:
 *
 * Proposal отмечается APPROVED
 * только ПОСЛЕ успешного сохранения Skill.
 *
 * Если Supabase вернул ошибку,
 * Proposal остаётся PENDING_APPROVAL,
 * чтобы сохранение можно было повторить.
 *
 * =========================================================
 */


/*
 * =========================================================
 * NORMALIZE ID
 * =========================================================
 */


function normalizeSkillId(
    value
) {

    return String(
        value || ""
    ).trim();

}


/*
 * =========================================================
 * GET VERSION FROM HISTORY ITEM
 * =========================================================
 */


function getHistoryVersion(
    item
) {

    const value =
        Number(
            item?.version ??
            item?.payload?.version
        );


    if (
        !Number.isInteger(
            value
        ) ||
        value < 1
    ) {

        return null;

    }


    return value;

}


/*
 * =========================================================
 * GET NEXT VERSION
 * =========================================================
 */


function getNextVersion(
    history
) {


    if (
        !Array.isArray(
            history
        ) ||
        history.length === 0
    ) {

        return 1;

    }


    const versions =
        history
            .map(
                getHistoryVersion
            )
            .filter(
                version =>
                    version !== null
            );


    if (
        versions.length === 0
    ) {

        return 1;

    }


    return (
        Math.max(
            ...versions
        ) + 1
    );

}


/*
 * =========================================================
 * FAILURE RESULT
 * =========================================================
 */


function createFailureResult({

    stage,

    error,

    proposal = null,

    skillId = null,

    version = null

} = {}) {


    return {

        success:
            false,

        stage:
            stage || "approval",

        proposal,

        experience:
            null,

        skillId,

        version,

        error:
            String(
                error ||
                "Не удалось подтвердить обучение"
            )

    };

}


/*
 * =========================================================
 * APPROVE AND SAVE
 * =========================================================
 */


export async function approveAndSaveLearningProposal({

    proposal,

    skillId = "",

    confidence = 0.7

} = {}) {


    /*
     * =====================================================
     * 1. PROPOSAL VALIDATION
     * =====================================================
     */


    if (
        !proposal ||
        typeof proposal !== "object" ||
        Array.isArray(
            proposal
        )
    ) {

        return createFailureResult({

            stage:
                "input",

            error:
                "Learning Proposal не указан"

        });

    }


    /*
     * Если Analyzer запросил уточнения,
     * сохранять Skill пока нельзя.
     */


    const clarificationQuestions =
        Array.isArray(
            proposal.clarificationQuestions
        )
            ? proposal.clarificationQuestions
            : [];


    if (
        clarificationQuestions.length > 0
    ) {

        return createFailureResult({

            stage:
                "clarification",

            proposal,

            error:
                "Learning Proposal требует уточнения перед сохранением"

        });

    }


    const proposedExperience =
        proposal.proposedExperience;


    if (
        !proposedExperience ||
        typeof proposedExperience !== "object" ||
        Array.isArray(
            proposedExperience
        )
    ) {

        return createFailureResult({

            stage:
                "input",

            proposal,

            error:
                "Learning Proposal не содержит proposedExperience"

        });

    }


    /*
     * =====================================================
     * 2. DETERMINE SKILL ID
     * =====================================================
     *
     * Приоритет:
     *
     * 1. skillId, переданный Approval явно;
     * 2. ID внутри proposedExperience;
     * 3. ID, автоматически созданный из name.
     *
     * Явный skillId понадобится,
     * когда пользователь исправляет
     * уже существующий Skill.
     *
     * =====================================================
     */


    const targetSkillId =
        normalizeSkillId(
            skillId
        ) ||
        normalizeSkillId(
            proposedExperience.id ||
            proposedExperience.skillId
        ) ||
        buildLearningSkillId(
            proposedExperience.name
        );


    if (!targetSkillId) {

        return createFailureResult({

            stage:
                "skill",

            proposal,

            error:
                "Не удалось определить Skill ID"

        });

    }


    /*
     * =====================================================
     * 3. LOAD HISTORY
     * =====================================================
     */


    let history;


    try {


        history =
            await getExperienceHistory(
                targetSkillId
            );


    } catch (error) {


        console.error(
            "Learning Approval history error:",
            error
        );


        return createFailureResult({

            stage:
                "history",

            proposal,

            skillId:
                targetSkillId,

            error:
                error?.message ||
                "Не удалось получить историю Skill"

        });

    }


    /*
     * =====================================================
     * 4. NEXT VERSION
     * =====================================================
     */


    const nextVersion =
        getNextVersion(
            history
        );


    /*
     * =====================================================
     * 5. BUILD SKILL
     * =====================================================
     */


    let experience;


    try {


        experience =
            buildExperienceSkill({

                proposedExperience,

                skillId:
                    targetSkillId,

                version:
                    nextVersion,

                confidence

            });


    } catch (error) {


        console.error(
            "Learning Approval Skill build error:",
            error
        );


        return createFailureResult({

            stage:
                "skill",

            proposal,

            skillId:
                targetSkillId,

            version:
                nextVersion,

            error:
                error?.message ||
                "Не удалось сформировать Experience Skill"

        });

    }


    /*
     * =====================================================
     * 6. ATOMIC SAVE
     * =====================================================
     *
     * Здесь используется:
     *
     * experienceStorage
     *      ↓
     * supabaseExperienceWriter
     *      ↓
     * PostgreSQL RPC
     *
     * History и Current сохраняются
     * одной транзакцией.
     *
     * =====================================================
     */


    let saveResult;


    try {


        saveResult =
            await saveExperienceSkill(
                experience
            );


    } catch (error) {


        console.error(
            "Learning Approval save error:",
            error
        );


        /*
         * Proposal НЕ переводим
         * в APPROVED.
         *
         * Его можно будет сохранить
         * повторно после устранения ошибки.
         */


        return createFailureResult({

            stage:
                "save",

            proposal,

            skillId:
                targetSkillId,

            version:
                nextVersion,

            error:
                error?.message ||
                "Не удалось сохранить Experience Skill"

        });

    }


    if (
        !saveResult?.success
    ) {

        return createFailureResult({

            stage:
                "save",

            proposal,

            skillId:
                targetSkillId,

            version:
                nextVersion,

            error:
                "Experience Storage не подтвердил сохранение Skill"

        });

    }


    /*
     * =====================================================
     * 7. APPROVE PROPOSAL
     * =====================================================
     *
     * Только теперь считаем обучение
     * успешно завершённым.
     *
     * =====================================================
     */


    let approvedProposal;


    try {


        approvedProposal =
            approveLearningProposal(
                proposal
            );


    } catch (error) {


        /*
         * Skill уже сохранён.
         *
         * Поэтому здесь нельзя утверждать,
         * что сохранение не произошло.
         */


        console.error(
            "Learning Proposal approval state error:",
            error
        );


        return {

            success:
                true,

            stage:
                "saved",

            proposal,

            proposalStateUpdated:
                false,

            experience,

            skillId:
                targetSkillId,

            version:
                nextVersion,

            previousVersions:
                Array.isArray(
                    history
                )
                    ? history.length
                    : 0,

            error:
                "Skill сохранён, но состояние Proposal не удалось обновить"

        };

    }


    /*
     * =====================================================
     * 8. SUCCESS
     * =====================================================
     */


    return {

        success:
            true,

        stage:
            "approved",

        proposal:
            approvedProposal,

        proposalStateUpdated:
            true,

        experience,

        skillId:
            targetSkillId,

        version:
            nextVersion,

        previousVersions:
            Array.isArray(
                history
            )
                ? history.length
                : 0,

        error:
            ""

    };

}
