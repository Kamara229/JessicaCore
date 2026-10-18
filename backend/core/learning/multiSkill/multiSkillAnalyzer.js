/*
 * =========================================================
 * JESSICA MULTI-SKILL ANALYZER
 * =========================================================
 *
 * Один successful complex Trace может реально
 * применить несколько Experience Skills.
 *
 * Каждый Skill получает собственный analysis
 * на основе своей child execution.
 * =========================================================
 */


import {
    analyzeExecutionTrace
} from "../experienceAnalyzer.js";


import {
    buildSkillTraceView
} from "./skillTraceView.js";


import {
    getLearningSkillId,
    uniqueLearningSkills
} from "./skillIdentity.js";


function getUsedSkills(
    trace
) {

    if(
        trace
            ?.experienceUsage
            ?.used !== true
    ){

        return [];

    }


    return uniqueLearningSkills(

        trace
            ?.experienceUsage
            ?.skills

    );

}


/*
 * =========================================================
 * PUBLIC
 * =========================================================
 */


export function analyzeExecutionTraceBatch(
    trace
) {

    const skills =

        getUsedSkills(
            trace
        );


    if(
        skills.length <= 1
    ){

        return [

            analyzeExecutionTrace(
                trace
            )

        ];

    }


    const result = [];


    for(
        const skill
        of skills
    ){

        const skillTrace =

            buildSkillTraceView(
                trace,
                skill
            );


        if(!skillTrace){

            continue;

        }


        const analysis =

            analyzeExecutionTrace(
                skillTrace
            );


        if(
            !analysis
            ||
            typeof analysis !== "object"
        ){

            continue;

        }


        result.push({

            ...analysis,

            metadata: {

                ...(analysis.metadata || {}),

                multiSkill:
                    true,

                targetSkillId:

                    analysis
                        ?.metadata
                        ?.targetSkillId

                    ||

                    getLearningSkillId(
                        skill
                    )

            }

        });

    }


    // Never reinforce a child Skill from parent-only execution evidence.
    return result;

}
