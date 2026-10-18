/*
 * =========================================================
 * JESSICA AUTONOMY DECISION
 * =========================================================
 */

import {
    APPROVAL_ACTION
} from "../approval/approvalConstants.js";


import {
    AUTONOMY_THRESHOLDS
} from "./autonomyThresholds.js";


import {
    evaluateSkillEvolution,
    EVOLUTION_MODE
} from "../quality/skillEvolutionGuard.js";


function keepCandidate(
    reason,
    metrics
) {

    return {

        action:
            APPROVAL_ACTION.KEEP_CANDIDATE,

        reason,

        metrics

    };

}


export function decideAutonomousLearning({

    proposal,

    action,

    metrics

}) {

    if(
        action ===
        "NEW_SKILL"
    ){

        if(
            metrics.confidence <
            AUTONOMY_THRESHOLDS.newSkillMinConfidence
        ){

            return keepCandidate(
                "Недостаточная уверенность для нового Skill",
                metrics
            );

        }


        return {

            action:
                APPROVAL_ACTION.AUTO_APPROVE,

            mode:
                "NEW_SKILL",

            reason:
                "Новый Skill прошёл Autonomy Policy",

            confidence:
                metrics.confidence,

            metrics

        };

    }


    if(
        action ===
        "SKILL_IMPROVEMENT"
    ){

        if(
            proposal
                ?.targetSkill
                ?.exists !== true
        ){

            return keepCandidate(
                "Existing Skill не определён",
                metrics
            );

        }


        if(
            metrics.confidence <
            AUTONOMY_THRESHOLDS.improvementMinConfidence
        ){

            return keepCandidate(
                "Недостаточная уверенность для изменения Skill",
                metrics
            );

        }


        const evolution =

            evaluateSkillEvolution(
                proposal
            );


        if(
            evolution.mode ===
            EVOLUTION_MODE.REINFORCEMENT_ONLY
        ){

            return {

                action:
                    APPROVAL_ACTION.REINFORCE_ONLY,

                mode:
                    "SKILL_REINFORCEMENT",

                reason:
                    evolution.reason,

                confidence:
                    metrics.confidence,

                metrics,

                evolution

            };

        }


        return {

            action:
                APPROVAL_ACTION.AUTO_APPROVE,

            mode:
                "SKILL_IMPROVEMENT",

            reason:
                evolution.reason,

            confidence:
                metrics.confidence,

            metrics,

            evolution

        };

    }


    return keepCandidate(
        "Неизвестный тип обучения",
        metrics
    );

}
