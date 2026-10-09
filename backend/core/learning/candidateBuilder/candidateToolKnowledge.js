/*
 * =========================================================
 * JESSICA CANDIDATE TOOL KNOWLEDGE
 * =========================================================
 *
 * Управляет машинным знанием
 * Experience о необходимых Tools.
 *
 *
 * NEW_SKILL:
 *
 * pattern.requiredTools
 *        +
 * trace.executedTools
 *
 *
 * SKILL_IMPROVEMENT:
 *
 * существующий requiredTools сохраняется.
 *
 * Если старый Skill имеет requiredTools=[],
 * возможен controlled bootstrap из
 * подтверждённого успешного Execution.
 *
 * =========================================================
 */


import {
    isObject,
    normalizeStringArray,
    normalizeText,
    mergeStringArrays,
    resolveObservedTools,
    resolveTraceSuccess
} from "./candidateUtils.js";


/*
 * =========================================================
 * NEW SKILL TOOLS
 * =========================================================
 */


export function resolveNewSkillRequiredTools({

    pattern,

    trace

}) {

    return mergeStringArrays(

        pattern?.requiredTools,

        resolveObservedTools(
            trace
        )

    );

}


/*
 * =========================================================
 * VERIFIED EXECUTION
 * =========================================================
 */


function isVerifiedLearningExecution(
    trace
) {

    if(
        !resolveTraceSuccess(
            trace
        )
    ){

        return false;

    }


    /*
     * Execution Result должен быть
     * реально верифицирован.
     */


    if(
        trace?.result?.verified !== true
    ){

        return false;

    }


    const validation =

        trace?.validation

        ||

        trace?.result?.validation

        ||

        null;


    if(
        !validation
        ||
        validation.valid !== true
    ){

        return false;

    }


    /*
     * no_verified_result не является
     * основанием для машинного bootstrap.
     */


    const outcomeType =

        normalizeText(
            validation.outcomeType
        );


    if(
        outcomeType
        &&
        outcomeType !== "result"
    ){

        return false;

    }


    return true;

}


/*
 * =========================================================
 * EXPERIENCE SELECTED
 * =========================================================
 */


function wasExperienceSelected(
    trace
) {

    const usage =

        isObject(
            trace?.experienceUsage
        )

            ? trace.experienceUsage

            : (
                isObject(
                    trace?.experience
                )

                    ? trace.experience

                    : {}
            );


    return (

        usage.found === true

        &&

        usage.used === true

    );

}


/*
 * =========================================================
 * IMPROVEMENT TOOLS
 * =========================================================
 *
 * Важное правило:
 *
 * non-empty requiredTools никогда
 * автоматически не расширяется.
 *
 *
 * Bootstrap разрешён только старым Skill,
 * которые появились до поддержки
 * executedTools и имеют requiredTools=[].
 *
 * =========================================================
 */


export function resolveImprovementRequiredTools({

    existingSkill,

    trace

}) {

    const existingTools =

        normalizeStringArray(
            existingSkill?.requiredTools
        );


    /*
     * Уже сформированный контракт
     * сохраняем как есть.
     */


    if(
        existingTools.length > 0
    ){

        return {

            requiredTools:
                existingTools,

            bootstrapped:
                false

        };

    }


    /*
     * Skill должен реально участвовать
     * в выполнении.
     */


    if(
        !wasExperienceSelected(
            trace
        )
    ){

        return {

            requiredTools:
                existingTools,

            bootstrapped:
                false

        };

    }


    /*
     * Execution должен быть
     * подтверждённым успешным результатом.
     */


    if(
        !isVerifiedLearningExecution(
            trace
        )
    ){

        return {

            requiredTools:
                existingTools,

            bootstrapped:
                false

        };

    }


    const observedTools =

        resolveObservedTools(
            trace
        );


    if(
        observedTools.length === 0
    ){

        return {

            requiredTools:
                existingTools,

            bootstrapped:
                false

        };

    }


    return {

        requiredTools:

            mergeStringArrays(

                existingTools,

                observedTools

            ),

        bootstrapped:
            true

    };

}
