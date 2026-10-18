/*
 * =========================================================
 * JESSICA SKILL QUALITY INSPECTOR v1
 * =========================================================
 */

import {
    getApprovalExperience
} from "../approval/approvalMetrics.js";


import {
    inspectSkillShape
} from "./inspectorShape.js";


import {
    inspectSkillEvidence
} from "./inspectorEvidence.js";


import {
    inspectExecutionProgram
} from "./inspectorProgram.js";


function mergeReports(
    reports
) {

    const errors = [];
    const warnings = [];
    const stats = {};


    for(
        const report
        of reports
    ){

        if(!report){

            continue;

        }


        errors.push(
            ...(
                Array.isArray(
                    report.errors
                )

                    ? report.errors

                    : []
            )
        );


        warnings.push(
            ...(
                Array.isArray(
                    report.warnings
                )

                    ? report.warnings

                    : []
            )
        );


        Object.assign(
            stats,
            report.stats || {}
        );

    }


    return {

        passed:
            errors.length === 0,

        errors,

        warnings,

        stats

    };

}


export function inspectLearningSkillQuality(
    proposal
) {

    const experience =

        getApprovalExperience(
            proposal
        );


    const report =

        mergeReports([

            inspectSkillShape(
                experience
            ),

            inspectSkillEvidence(
                experience,
                proposal
            ),

            inspectExecutionProgram(
                experience
            )

        ]);


    return {

        ...report,

        reason:

            report.passed

                ? "Skill прошёл Quality Inspector"

                : report.errors.join("; ")

    };

}
