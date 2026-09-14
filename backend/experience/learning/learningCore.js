import {
    buildLearningProposal
} from "./learningProposalBuilder.js";

import {
    approveAndSaveLearningProposal
} from "./learningApproval.js";

import {
    rejectLearningProposal
} from "./learningProposal.js";


/*
 * =========================================================
 * JESSICA LEARNING CORE
 * =========================================================
 *
 * Центральная точка доступа
 * ко всей системе обучения Jessica.
 *
 *
 * Основная цепочка:
 *
 * пользователь исправляет Jessica
 *          ↓
 * createLearning()
 *          ↓
 * Learning Analyzer
 *          ↓
 * Learning Proposal
 *          ↓
 * PENDING_APPROVAL
 *          ↓
 * пользователь подтверждает / отклоняет
 *          ↓
 * approveLearning() / rejectLearning()
 *
 *
 * Только approveLearning()
 * может привести к сохранению Skill.
 *
 *
 * Этот файл НЕ содержит:
 *
 * - AI prompts;
 * - JSON parsing;
 * - Supabase-запросы;
 * - построение Skill;
 * - алгоритм версионирования;
 * - Experience Search;
 * - Planner.
 *
 * =========================================================
 */


/*
 * =========================================================
 * CREATE LEARNING
 * =========================================================
 *
 * Анализирует исправление пользователя
 * и создаёт Learning Proposal.
 *
 * Ничего не сохраняет в Experience.
 *
 * =========================================================
 */


export async function createLearning({

    task,

    previousAnswer = "",

    correction,

    correctedAnswer = ""

} = {}) {


    try {


        const result =
            await buildLearningProposal({

                task,

                previousAnswer,

                correction,

                correctedAnswer

            });


        return result;


    } catch (error) {


        console.error(
            "Learning Core create error:",
            error
        );


        return {

            success:
                false,

            stage:
                "learning",

            proposal:
                null,

            analysis:
                null,

            reusable:
                false,

            needsClarification:
                false,

            readyForApproval:
                false,

            error:
                error?.message ||
                "Не удалось создать предложение обучения"

        };


    }

}


/*
 * =========================================================
 * APPROVE LEARNING
 * =========================================================
 *
 * Принимает именно результат createLearning(),
 * а не произвольный Proposal.
 *
 *
 * Это позволяет проверить:
 *
 * readyForApproval === true
 *
 * перед сохранением Skill.
 *
 * =========================================================
 */


export async function approveLearning({

    learningResult,

    skillId = "",

    confidence = 0.7

} = {}) {


    /*
     * =====================================================
     * INPUT
     * =====================================================
     */


    if (
        !learningResult ||
        typeof learningResult !== "object"
    ) {

        return {

            success:
                false,

            stage:
                "input",

            proposal:
                null,

            experience:
                null,

            error:
                "Learning Result не указан"

        };

    }


    if (
        learningResult.success !== true
    ) {

        return {

            success:
                false,

            stage:
                "input",

            proposal:
                learningResult.proposal || null,

            experience:
                null,

            error:
                "Нельзя подтвердить неуспешный Learning Result"

        };

    }


    /*
     * =====================================================
     * CLARIFICATION
     * =====================================================
     */


    if (
        learningResult.needsClarification === true
    ) {

        return {

            success:
                false,

            stage:
                "clarification",

            proposal:
                learningResult.proposal || null,

            experience:
                null,

            error:
                "Перед подтверждением обучения нужны уточнения пользователя"

        };

    }


    /*
     * =====================================================
     * REUSABLE CHECK
     * =====================================================
     */


    if (
        learningResult.reusable !== true
    ) {

        return {

            success:
                false,

            stage:
                "reusable",

            proposal:
                learningResult.proposal || null,

            experience:
                null,

            error:
                "Исправление не признано переиспользуемым Skill"

        };

    }


    /*
     * =====================================================
     * APPROVAL READINESS
     * =====================================================
     */


    if (
        learningResult.readyForApproval !== true
    ) {

        return {

            success:
                false,

            stage:
                "approval",

            proposal:
                learningResult.proposal || null,

            experience:
                null,

            error:
                "Learning Proposal ещё не готов к подтверждению"

        };

    }


    const proposal =
        learningResult.proposal;


    if (!proposal) {

        return {

            success:
                false,

            stage:
                "proposal",

            proposal:
                null,

            experience:
                null,

            error:
                "Learning Proposal отсутствует"

        };

    }


    /*
     * =====================================================
     * SAVE APPROVED EXPERIENCE
     * =====================================================
     */


    try {


        return await approveAndSaveLearningProposal({

            proposal,

            skillId,

            confidence

        });


    } catch (error) {


        console.error(
            "Learning Core approval error:",
            error
        );


        return {

            success:
                false,

            stage:
                "approval",

            proposal,

            experience:
                null,

            error:
                error?.message ||
                "Не удалось подтвердить обучение"

        };


    }

}


/*
 * =========================================================
 * REJECT LEARNING
 * =========================================================
 *
 * Отклонение Proposal никогда
 * не изменяет Experience.
 *
 * =========================================================
 */


export function rejectLearning({

    learningResult

} = {}) {


    if (
        !learningResult ||
        typeof learningResult !== "object"
    ) {

        return {

            success:
                false,

            stage:
                "input",

            proposal:
                null,

            error:
                "Learning Result не указан"

        };

    }


    const proposal =
        learningResult.proposal;


    if (!proposal) {

        return {

            success:
                false,

            stage:
                "proposal",

            proposal:
                null,

            error:
                "Learning Proposal отсутствует"

        };

    }


    try {


        const rejectedProposal =
            rejectLearningProposal(
                proposal
            );


        return {

            success:
                true,

            stage:
                "rejected",

            proposal:
                rejectedProposal,

            error:
                ""

        };


    } catch (error) {


        console.error(
            "Learning Core reject error:",
            error
        );


        return {

            success:
                false,

            stage:
                "reject",

            proposal,

            error:
                error?.message ||
                "Не удалось отклонить Learning Proposal"

        };


    }

}
