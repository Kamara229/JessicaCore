/*
 * =========================================================
 * JESSICA SKILL IMPROVEMENT CANDIDATE FACADE v5
 * =========================================================
 *
 * Existing Skill + new verified execution
 *      ↓
 * context builder
 *      ↓
 * result builder
 * =========================================================
 */


import {
    buildSkillImprovementContext
} from "./skillImprovement/skillImprovementContext.js";


import {
    buildSkillImprovementResult
} from "./skillImprovement/skillImprovementResult.js";


export function buildSkillImprovementCandidate({

    skills,

    trace,

    confidence,

    maturity,

    occurrences

}) {

    const context =

        buildSkillImprovementContext({

            skills,

            trace,

            confidence,

            maturity,

            occurrences

        });


    return buildSkillImprovementResult(
        context
    );

}
