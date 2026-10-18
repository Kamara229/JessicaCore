/*
 * =========================================================
 * JESSICA SKILL TRACE VIEW FACADE
 * =========================================================
 */


import {
    getLearningSkillId
} from "./skillIdentity.js";


import {
    resultUsesSkill,
    safeLearningArray
} from "./skillTraceResult.js";


import {
    buildChildSkillTrace
} from "./skillTraceBuilder.js";


export function buildSkillTraceView(
    trace,
    skill
) {

    const skillId =

        getLearningSkillId(
            skill
        );


    if(
        !trace
        ||
        !skillId
    ){

        return null;

    }


    const childResults =

        safeLearningArray(
            trace
                ?.result
                ?.results
        );


    const matched =

        childResults.filter(
            result =>
                resultUsesSkill(
                    result,
                    skillId
                )
        );


    if(
        matched.length > 0
    ){

        return buildChildSkillTrace(

            trace,

            matched[
                matched.length - 1
            ],

            skill

        );

    }


    // Absent child evidence must not inherit the parent's successful result.
    return null;
}
