import {
    randomUUID
} from "node:crypto";


/*
 * =========================================================
 * JESSICA LEARNING PROPOSAL
 * =========================================================
 *
 * Модель предложения обучения Jessica.
 *
 *
 * ВАЖНО:
 *
 * Learning Proposal ещё НЕ является Skill.
 *
 * Это черновик нового опыта,
 * который должен быть подтверждён
 * пользователем перед сохранением.
 *
 *
 * Жизненный цикл:
 *
 * PENDING_APPROVAL
 *        ↓
 * APPROVED
 *        ↓
 * сохранение Skill
 *
 *
 * Или:
 *
 * PENDING_APPROVAL
 *        ↓
 * REJECTED
 *
 *
 * Этот модуль НЕ:
 *
 * - вызывает AI;
 * - анализирует ошибку;
 * - сохраняет Skill;
 * - работает с Supabase;
 * - вызывает Planner;
 * - выполняет задачу.
 *
 * =========================================================
 */


export const LEARNING_PROPOSAL_STATUS = {

    PENDING_APPROVAL:
        "PENDING_APPROVAL",

    APPROVED:
        "APPROVED",

    REJECTED:
        "REJECTED"

};


/*
 * =========================================================
 * NORMALIZE TEXT
 * =========================================================
 */


function normalizeText(
    value
) {

    return String(
        value || ""
    ).trim();

}


/*
 * =========================================================
 * NORMALIZE ARRAY
 * =========================================================
 */


function normalizeTextArray(
    value
) {

    if (
        !Array.isArray(
            value
        )
    ) {

        return [];

    }


    return value
        .map(
            item =>
                normalizeText(
                    item
                )
        )
        .filter(
            Boolean
        );

}


/*
 * =========================================================
 * CREATE LEARNING PROPOSAL
 * =========================================================
 */


export function createLearningProposal({

    task,

    previousAnswer = "",

    correction = "",

    correctedAnswer = "",

    understanding = "",

    clarificationQuestions = [],

    proposedExperience = null

} = {}) {


    const cleanTask =
        normalizeText(
            task
        );


    const cleanCorrection =
        normalizeText(
            correction
        );


    if (!cleanTask) {

        throw new Error(
            "Learning Proposal: исходная задача не указана"
        );

    }


    if (!cleanCorrection) {

        throw new Error(
            "Learning Proposal: исправление пользователя не указано"
        );

    }


    return {

        id:
            randomUUID(),

        status:
            LEARNING_PROPOSAL_STATUS
                .PENDING_APPROVAL,

        task:
            cleanTask,

        previousAnswer:
            normalizeText(
                previousAnswer
            ),

        correction:
            cleanCorrection,

        correctedAnswer:
            normalizeText(
                correctedAnswer
            ),

        understanding:
            normalizeText(
                understanding
            ),

        clarificationQuestions:
            normalizeTextArray(
                clarificationQuestions
            ),

        proposedExperience:
            proposedExperience &&
            typeof proposedExperience === "object"
                ? proposedExperience
                : null,

        createdAt:
            new Date()
                .toISOString(),

        approvedAt:
            null,

        rejectedAt:
            null

    };

}


/*
 * =========================================================
 * APPROVE PROPOSAL
 * =========================================================
 *
 * Пока только меняет состояние объекта.
 *
 * Сохранение Skill будет выполняться
 * отдельным модулем Learning.
 *
 * =========================================================
 */


export function approveLearningProposal(
    proposal
) {


    if (
        !proposal ||
        typeof proposal !== "object"
    ) {

        throw new Error(
            "Learning Proposal не указан"
        );

    }


    if (
        proposal.status !==
        LEARNING_PROPOSAL_STATUS
            .PENDING_APPROVAL
    ) {

        throw new Error(
            "Learning Proposal уже обработан"
        );

    }


    return {

        ...proposal,

        status:
            LEARNING_PROPOSAL_STATUS
                .APPROVED,

        approvedAt:
            new Date()
                .toISOString(),

        rejectedAt:
            null

    };

}


/*
 * =========================================================
 * REJECT PROPOSAL
 * =========================================================
 */


export function rejectLearningProposal(
    proposal
) {


    if (
        !proposal ||
        typeof proposal !== "object"
    ) {

        throw new Error(
            "Learning Proposal не указан"
        );

    }


    if (
        proposal.status !==
        LEARNING_PROPOSAL_STATUS
            .PENDING_APPROVAL
    ) {

        throw new Error(
            "Learning Proposal уже обработан"
        );

    }


    return {

        ...proposal,

        status:
            LEARNING_PROPOSAL_STATUS
                .REJECTED,

        approvedAt:
            null,

        rejectedAt:
            new Date()
                .toISOString()

    };

}
