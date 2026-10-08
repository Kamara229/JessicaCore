/*
 * =========================================================
 * JESSICA TRACE RESULT v3
 * =========================================================
 *
 * Синхронизация Execution Result
 * с внешним Execution Trace.
 *
 *
 * Ответственность:
 *
 * - Single Execution Result;
 * - Complex Execution Summary;
 * - Validation;
 * - Terminal;
 * - Failure metadata;
 * - Experience usage;
 * - Execution statistics;
 * - фактически выполненные Tools.
 *
 *
 * КРИТИЧНО:
 *
 * Planner Plan и фактический Execution —
 * разные сущности.
 *
 * Поэтому executedTools собираются
 * только из реальных Tool Result.
 *
 * =========================================================
 */


import {
    addTraceEvent
} from "./traceEvents.js";


import {
    collectTraceExecutedTools
} from "./traceToolEvidence.js";


/*
 * =========================================================
 * HELPERS
 * =========================================================
 */


function safeArray(
    value
) {

    return Array.isArray(value)

        ? value

        : [];

}


function safeNumber(
    value
) {

    const number =
        Number(value);


    return Number.isFinite(number)

        ? number

        : 0;

}


function isObject(
    value
) {

    return Boolean(

        value
        &&
        typeof value === "object"
        &&
        !Array.isArray(value)

    );

}


/*
 * =========================================================
 * SKILL ID
 * =========================================================
 */


function getSkillId(
    skill
) {

    if(
        typeof skill === "string"
    ){

        return skill.trim();

    }


    return String(

        skill?.id

        ||

        skill?.skillId

        ||

        skill?.name

        ||

        ""

    )
    .trim();

}


/*
 * =========================================================
 * EXPERIENCE SOURCE
 * =========================================================
 */


function resolveResultExperience(
    result
) {

    /*
     * Новый канонический контракт
     * executeSubtask().
     */


    if(
        isObject(
            result?.experience
        )
    ){

        return result.experience;

    }


    /*
     * Старый compatibility contract.
     */


    if(
        isObject(
            result?.executionMeta?.experience
        )
    ){

        return result.executionMeta.experience;

    }


    /*
     * Fallback для flattened executionMeta.
     */


    const meta =
        result?.executionMeta;


    if(
        isObject(meta)
        &&
        (
            meta.experienceUsed !== undefined
            ||
            meta.experienceFound !== undefined
            ||
            meta.experienceSource !== undefined
        )
    ){

        return {

            used:

                meta.experienceUsed === true,


            found:

                meta.experienceFound === true,


            source:

                meta.experienceSource

                ||

                null,


            confidence:

                safeNumber(
                    meta.experienceConfidence
                ),


            skills:

                []

        };

    }


    return null;

}


/*
 * =========================================================
 * EXPERIENCE
 * =========================================================
 */


function collectExperience(
    trace,
    result
) {

    const experience =

        resolveResultExperience(
            result
        );


    if(
        !experience
    ){

        return;

    }


    if(
        !isObject(
            trace.experienceUsage
        )
    ){

        trace.experienceUsage = {

            used:

                false,


            found:

                false,


            source:

                null,


            confidence:

                0,


            skills:

                [],


            skillIds:

                []

        };

    }


    if(
        !Array.isArray(
            trace.experienceUsage.skills
        )
    ){

        trace.experienceUsage.skills = [];

    }


    if(
        !Array.isArray(
            trace.experienceUsage.skillIds
        )
    ){

        trace.experienceUsage.skillIds = [];

    }


    /*
     * Experience считается найденным,
     * если хотя бы один Child Execution
     * получил подходящий Skill.
     */


    trace.experienceUsage.found =

        trace.experienceUsage.found === true

        ||

        experience.found === true;


    /*
     * Compatibility field.
     *
     * Пока означает:
     * Experience был передан/использован
     * Execution Layer.
     *
     * Позже введём отдельный applied.
     */


    trace.experienceUsage.used =

        trace.experienceUsage.used === true

        ||

        experience.used === true;


    /*
     * Source.
     */


    if(
        experience.source
    ){

        trace.experienceUsage.source =
            experience.source;

    }


    /*
     * Для агрегированного Trace
     * сохраняем максимальную confidence.
     */


    trace.experienceUsage.confidence =

        Math.max(

            safeNumber(
                trace.experienceUsage.confidence
            ),

            safeNumber(
                experience.confidence
            )

        );


    const skills =

        safeArray(
            experience.skills
        );


    for(
        const skill
        of skills
    ){

        const id =

            getSkillId(
                skill
            );


        if(
            !id
        ){

            continue;

        }


        if(
            trace.experienceUsage
                .skillIds
                .includes(id)
        ){

            continue;

        }


        trace.experienceUsage
            .skillIds
            .push(id);


        trace.experienceUsage
            .skills
            .push(

                isObject(skill)

                    ? {
                        ...skill
                    }

                    : skill

            );

    }

}


/*
 * =========================================================
 * EXECUTION META
 * =========================================================
 */


function collectMeta(
    trace,
    result
) {

    const meta =

        result?.executionMeta;


    if(
        !isObject(meta)
    ){

        return;

    }


    if(
        !isObject(
            trace.statistics
        )
    ){

        trace.statistics = {

            attempts:

                0,


            retries:

                0,


            replans:

                0,


            completed:

                0,


            failed:

                0

        };

    }


    /*
     * Compatibility:
     *
     * некоторые Execution Result
     * передают retryCount/replanCount.
     */


    if(
        meta.retryCount !== undefined
    ){

        trace.statistics.retries =

            Math.max(

                safeNumber(
                    trace.statistics.retries
                ),

                safeNumber(
                    meta.retryCount
                )

            );

    }


    if(
        meta.replanCount !== undefined
    ){

        trace.statistics.replans =

            Math.max(

                safeNumber(
                    trace.statistics.replans
                ),

                safeNumber(
                    meta.replanCount
                )

            );

    }


    if(
        meta.attemptCount !== undefined
    ){

        trace.statistics.attempts =

            Math.max(

                safeNumber(
                    trace.statistics.attempts
                ),

                safeNumber(
                    meta.attemptCount
                )

            );

    }

}


/*
 * =========================================================
 * FAILURE
 * =========================================================
 */


function collectFailure(
    trace,
    result
) {

    if(
        !result?.failure
    ){

        return;

    }


    if(
        !Array.isArray(
            trace.failures
        )
    ){

        trace.failures = [];

    }


    trace.failures.push({

        ...result.failure,


        timestamp:

            new Date()
                .toISOString()

    });

}


/*
 * =========================================================
 * SUCCESS
 * =========================================================
 */


function isSuccessfulResult(
    result
) {

    if(
        result?.success === true
    ){

        return true;

    }


    return (

        String(
            result?.status || ""
        )
        .trim()
        .toUpperCase()

        ===

        "COMPLETED"

    );

}


/*
 * =========================================================
 * OUTCOME STATISTICS
 * =========================================================
 */


function setOutcomeStatistics(
    trace,
    results
) {

    const safeResults =

        safeArray(
            results
        );


    const completed =

        safeResults
            .filter(
                isSuccessfulResult
            )
            .length;


    const failed =

        safeResults.length
        -
        completed;


    trace.statistics = {

        ...(trace.statistics || {}),


        completed,


        failed

    };


    /*
     * Compatibility alias.
     *
     * Старые Learning-модули
     * ещё могут читать trace.stats.
     *
     * Канонический источник:
     *
     * trace.statistics.
     */


    trace.stats = {

        ...(trace.stats || {}),


        completed,


        failed,


        attempts:

            safeNumber(
                trace.statistics.attempts
            ),


        retries:

            safeNumber(
                trace.statistics.retries
            ),


        replans:

            safeNumber(
                trace.statistics.replans
            )

    };

}


/*
 * =========================================================
 * SINGLE RESULT
 * =========================================================
 */


export function updateTraceFromResult(
    trace,
    result
) {

    if(
        !trace
        ||
        !result
    ){

        return trace;

    }


    /*
     * =====================================================
     * RESULT
     * =====================================================
     */


    trace.result = {

        ...result

    };


    trace.validation =

        result.validation

        ||

        null;


    trace.terminal =

        result.terminal

        ||

        null;


    /*
     * =====================================================
     * EVENT
     * =====================================================
     */


    addTraceEvent(

        trace,

        "RESULT",

        {

            status:

                result.status,


            success:

                result.success,


            verified:

                result.verified

        }

    );


    /*
     * =====================================================
     * FAILURE
     * =====================================================
     */


    collectFailure(
        trace,
        result
    );


    /*
     * =====================================================
     * EXPERIENCE
     * =====================================================
     */


    collectExperience(
        trace,
        result
    );


    /*
     * =====================================================
     * META
     * =====================================================
     */


    collectMeta(
        trace,
        result
    );


    /*
     * =====================================================
     * ACTUAL EXECUTED TOOLS
     * =====================================================
     */


    collectTraceExecutedTools(
        trace,
        result
    );


    /*
     * =====================================================
     * OUTCOME
     * =====================================================
     */


    setOutcomeStatistics(

        trace,

        [
            result
        ]

    );


    return trace;

}


/*
 * =========================================================
 * COMPLEX SUMMARY
 * =========================================================
 */


export function updateTraceFromSummary(
    trace,
    summary
) {

    if(
        !trace
        ||
        !summary
    ){

        return trace;

    }


    /*
     * =====================================================
     * SUMMARY RESULT
     * =====================================================
     */


    trace.result = {

        ...summary

    };


    trace.validation =

        summary.validation

        ||

        null;


    trace.terminal =

        summary.terminal

        ||

        null;


    const results =

        safeArray(
            summary.results
        );


    /*
     * =====================================================
     * EVENT
     * =====================================================
     */


    addTraceEvent(

        trace,

        "SUMMARY",

        {

            status:

                summary.status

                ||

                null,


            success:

                summary.success === true,


            resultsCount:

                results.length

        }

    );


    /*
     * =====================================================
     * SUMMARY META
     * =====================================================
     */


    collectFailure(
        trace,
        summary
    );


    collectExperience(
        trace,
        summary
    );


    collectMeta(
        trace,
        summary
    );


    /*
     * Summary иногда уже содержит
     * вложенные Tool Results.
     */


    collectTraceExecutedTools(
        trace,
        summary
    );


    /*
     * =====================================================
     * CHILD RESULTS
     * =====================================================
     */


    for(
        const result
        of results
    ){

        if(
            !isObject(result)
        ){

            continue;

        }


        collectFailure(
            trace,
            result
        );


        collectExperience(
            trace,
            result
        );


        collectMeta(
            trace,
            result
        );


        collectTraceExecutedTools(
            trace,
            result
        );

    }


    /*
     * =====================================================
     * OUTCOME
     * =====================================================
     *
     * Если есть Child Results,
     * именно они являются источником
     * completed / failed.
     *
     * Если их нет —
     * используем Summary.
     *
     * =====================================================
     */


    setOutcomeStatistics(

        trace,

        results.length > 0

            ? results

            : [
                summary
            ]

    );


    return trace;

}
