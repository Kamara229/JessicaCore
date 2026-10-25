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
import {missingNativeDateExplanation} from "../nativeDateExplanationEvidence.js";


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
        || trace?.experienceUsage?.applied !== true
    ){

        return [];

    }


    const appliedIds=new Set(Array.isArray(trace?.experienceUsage?.appliedSkillIds)
        ? trace.experienceUsage.appliedSkillIds : []);
    return uniqueLearningSkills(trace?.experienceUsage?.skills)
        .filter(skill=>appliedIds.has(getLearningSkillId(skill)));

}


/*
 * =========================================================
 * PUBLIC
 * =========================================================
 */


export function analyzeExecutionTraceBatch(
    trace
) {
    if(missingNativeDateExplanation(trace))
        return [{action:"IGNORE",reusable:false,reason:"Ответ не объясняет запрошенный расчёт",
          analysisType:"INCOMPLETE_DATE_EXPLANATION",
          metadata:{positiveReinforcementAllowed:false}}];


    const skills =

        getUsedSkills(
            trace
        );


    // Even one applied Skill in a complex parent trace needs its own
    // child verdict. Parent success must never override a failed child.
    if(
        skills.length === 0 ||
        !Array.isArray(trace?.result?.results)
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
