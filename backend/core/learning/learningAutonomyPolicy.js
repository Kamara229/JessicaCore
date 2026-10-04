/*
 * =========================================================
 * JESSICA LEARNING AUTONOMY POLICY v4
 * =========================================================
 *
 * Решает:
 *
 * достаточно ли подтверждений,
 * чтобы Jessica самостоятельно
 * сохранила Experience Skill.
 *
 *
 * Возможные решения:
 *
 * AUTO_APPROVE
 * KEEP_CANDIDATE
 *
 *
 * НЕ:
 *
 * - выполняет structural validation;
 * - сохраняет Skill;
 * - работает с БД;
 * - вызывает AI.
 *
 * =========================================================
 */


import {
    getApprovalExperience,
    getApprovalMetrics
} from "./approval/approvalMetrics.js";


import {
    APPROVAL_ACTION
} from "./approval/approvalConstants.js";


const NEW_SKILL_MIN_CONFIDENCE =
    0.80;


const IMPROVEMENT_MIN_CONFIDENCE =
    0.60;


const MIN_SUCCESS_RATE =
    0.80;


const MIN_MATURITY =
    0.50;


const MIN_EXAMPLES =
    1;


const MIN_OCCURRENCES =
    1;


function keepCandidate(
    reason,
    metrics = null
) {

    return {

        action:
            APPROVAL_ACTION.KEEP_CANDIDATE,

        reason,

        metrics

    };

}


export function evaluateLearningAutonomy(
    proposal
) {


    if(
        !proposal ||
        typeof proposal !== "object"
    ){

        return keepCandidate(
            "Proposal отсутствует"
        );

    }


    if(
        proposal?.analysis?.reusable !== true
    ){

        return keepCandidate(
            "Experience не признан reusable"
        );

    }


    const action =
        proposal.action;


    const experience =

        getApprovalExperience(
            proposal
        );


    const metrics =

        getApprovalMetrics(
            proposal
        );


    if(
        !experience.workflow?.length
    ){

        return keepCandidate(
            "Workflow отсутствует",
            metrics
        );

    }


    if(
        metrics.examplesCount <
        MIN_EXAMPLES
    ){

        return keepCandidate(
            "Недостаточно подтверждённых Examples",
            metrics
        );

    }


    if(
        !Number.isFinite(
            metrics.occurrences
        )
        ||
        metrics.occurrences <
        MIN_OCCURRENCES
    ){

        return keepCandidate(
            "Недостаточно наблюдений",
            metrics
        );

    }


    /*
     * Отсутствующая метрика больше
     * не означает разрешение обучаться.
     */


    if(
        !Number.isFinite(
            metrics.successRate
        )
    ){

        return keepCandidate(
            "Success Rate отсутствует",
            metrics
        );

    }


    if(
        metrics.successRate <
        MIN_SUCCESS_RATE
    ){

        return keepCandidate(
            "Недостаточный показатель успешности",
            metrics
        );

    }


    if(
        !Number.isFinite(
            metrics.maturity
        )
    ){

        return keepCandidate(
            "Maturity отсутствует",
            metrics
        );

    }


    if(
        metrics.maturity <
        MIN_MATURITY
    ){

        return keepCandidate(
            "Недостаточная зрелость Experience",
            metrics
        );

    }


    if(
        !Number.isFinite(
            metrics.confidence
        )
    ){

        return keepCandidate(
            "Confidence отсутствует",
            metrics
        );

    }


    /*
     * =====================================================
     * NEW SKILL
     * =====================================================
     */


    if(
        action ===
        "NEW_SKILL"
    ){

        if(
            metrics.confidence <
            NEW_SKILL_MIN_CONFIDENCE
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


    /*
     * =====================================================
     * SKILL IMPROVEMENT
     * =====================================================
     */


    if(
        action ===
        "SKILL_IMPROVEMENT"
    ){

        if(
            proposal?.targetSkill?.exists !== true
        ){

            return keepCandidate(

                "Existing Skill не определён",

                metrics

            );

        }


        if(
            metrics.confidence <
            IMPROVEMENT_MIN_CONFIDENCE
        ){

            return keepCandidate(

                "Недостаточная уверенность для изменения Skill",

                metrics

            );

        }


        return {

            action:
                APPROVAL_ACTION.AUTO_APPROVE,

            mode:
                "SKILL_IMPROVEMENT",

            reason:
                "Existing Skill прошёл Autonomy Policy для новой версии",

            confidence:
                metrics.confidence,

            metrics

        };

    }


    return keepCandidate(

        "Неизвестный тип обучения",

        metrics

    );

}
