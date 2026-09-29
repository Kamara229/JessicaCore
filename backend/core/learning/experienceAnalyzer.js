/*
 * =========================================================
 * JESSICA EXPERIENCE ANALYZER v4
 * =========================================================
 *
 * Главный координатор анализа опыта.
 *
 *
 * Flow:
 *
 * Execution Trace
 *        ↓
 * Experience Analyzer
 *
 *        ↓
 * Pattern Matcher
 *
 *        ↓
 * Confidence Calculator
 *
 *        ↓
 * Candidate Builder
 *
 *        ↓
 * Learning Candidate
 *
 *
 * Ответственность:
 *
 * - принять Execution Trace;
 * - определить тип обучения;
 * - собрать результат анализа.
 *
 *
 * НЕ:
 *
 * - хранит паттерны;
 * - считает confidence;
 * - создаёт Skill;
 * - пишет в память.
 *
 * =========================================================
 */



import {
    matchExperiencePattern
} from "./experiencePatternMatcher.js";


import {
    getOccurrences,
    getExamples,
    calculateSuccessRate,
    calculateExperienceConfidence,
    calculateExperienceMaturity
} from "./experienceConfidence.js";


import {
    buildNewSkillCandidate,
    buildSkillImprovementCandidate
} from "./experienceCandidateBuilder.js";









/*
 * =========================================================
 * VALIDATE TRACE
 * =========================================================
 */


function isValidTrace(
    trace
) {

    return (

        trace &&

        typeof trace === "object"

    );

}









/*
 * =========================================================
 * ANALYZE EXISTING SKILL
 * =========================================================
 */


function analyzeExistingSkill(
    trace
) {


    const usage =
        trace?.experienceUsage;



    if(
        !usage ||
        usage.used !== true
    ){

        return null;

    }




    if(
        !Array.isArray(
            usage.skills
        )
        ||
        usage.skills.length === 0
    ){

        return null;

    }




    const occurrences =

        getOccurrences(
            trace
        );



    const examples =

        getExamples(
            trace
        );



    const successRate =

        calculateSuccessRate(
            examples
        );



    const confidence =

        calculateExperienceConfidence({

            matchScore:
                1,


            successRate,


            occurrences

        });



    const maturity =

        calculateExperienceMaturity(
            occurrences
        );







    return {


        action:

            "SKILL_IMPROVEMENT",



        reusable:

            true,



        reason:

            "Существующий Skill получил успешный новый опыт",



        skillCandidate:

            buildSkillImprovementCandidate({

                skills:
                    usage.skills,


                trace,


                confidence,


                maturity,


                occurrences

            })

    };


}









/*
 * =========================================================
 * ANALYZE NEW SKILL
 * =========================================================
 */


function analyzeNewSkill(
    trace
) {


    const matched =

        matchExperiencePattern(
            trace.task
        );



    if(
        !matched
    ){

        return null;

    }





    const occurrences =

        getOccurrences(
            trace
        );



    const examples =

        getExamples(
            trace
        );



    const successRate =

        calculateSuccessRate(
            examples
        );



    const confidence =

        calculateExperienceConfidence({

            matchScore:

                matched.matchScore,


            successRate,


            occurrences

        });



    const maturity =

        calculateExperienceMaturity(
            occurrences
        );






    return {


        action:

            "NEW_SKILL",



        reusable:

            true,



        reason:

            "Обнаружен повторяемый сценарий, пригодный для Skill",



        skillCandidate:

            buildNewSkillCandidate({

                pattern:

                    matched.pattern,


                trace,


                confidence,


                maturity,


                occurrences

            })

    };


}









/*
 * =========================================================
 * MAIN ANALYSIS
 * =========================================================
 */


export function analyzeExecutionTrace(
    trace
) {


    if(
        !isValidTrace(
            trace
        )
    ){

        return {


            action:

                "IGNORE",



            reusable:

                false,



            reason:

                "Execution Trace отсутствует",



            skillCandidate:

                null

        };

    }







    /*
     * Анализируем только успешные выполнения
     */


    if(
        (trace.stats?.completed || 0) <= 0
    ){

        return {


            action:

                "IGNORE",



            reusable:

                false,



            reason:

                "Нет успешного выполнения",



            skillCandidate:

                null

        };

    }









    /*
     * 1. Улучшение существующего Skill
     */


    const existing =

        analyzeExistingSkill(
            trace
        );



    if(
        existing
    ){

        return existing;

    }









    /*
     * 2. Новый Skill
     */


    const newSkill =

        analyzeNewSkill(
            trace
        );



    if(
        newSkill
    ){

        return newSkill;

    }









    /*
     * 3. Нечему учиться
     */


    return {


        action:

            "IGNORE",



        reusable:

            false,



        reason:

            "Повторяемый сценарий не найден",



        skillCandidate:

            null

    };


}
