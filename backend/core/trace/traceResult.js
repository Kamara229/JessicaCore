/*
 * =========================================================
 * JESSICA TRACE RESULT v1
 * =========================================================
 *
 * Синхронизация Execution Result с Trace.
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
 * - Execution statistics.
 *
 *
 * НЕ:
 *
 * - создаёт Trace;
 * - выполняет Execution;
 * - принимает решения;
 * - завершает Trace.
 *
 * =========================================================
 */


import {
    addTraceEvent
} from "./traceEvents.js";



/*
 * =========================================================
 * SAFE ARRAY
 * =========================================================
 */


function safeArray(

    value

){


    return Array.isArray(
        value
    )
        ?
        value
        :
        [];

}



/*
 * =========================================================
 * SKILL ID
 * =========================================================
 */


function getSkillId(

    skill

){


    if(
        typeof skill === "string"
    ){

        return skill;

    }



    return (

        skill?.id ||

        skill?.name ||

        ""

    );

}



/*
 * =========================================================
 * EXPERIENCE
 * =========================================================
 */


function collectExperience(

    trace,

    result

){


    const experience =

        result?.executionMeta?.experience;



    if(!experience)
        return;



    if(
        !trace.experienceUsage ||
        typeof trace.experienceUsage !== "object"
    ){

        trace.experienceUsage = {

            used:false,

            source:null,

            confidence:0,

            skills:[],

            skillIds:[]

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



    trace.experienceUsage.used =

        trace.experienceUsage.used

        ||

        experience.used === true;



    trace.experienceUsage.source =

        experience.source ||

        trace.experienceUsage.source;



    trace.experienceUsage.confidence =

        Number(

            experience.confidence || 0

        );



    const skills =

        safeArray(

            experience.skills

        );



    for(
        const skill of skills
    ){


        const id =

            getSkillId(
                skill
            );



        if(!id)
            continue;



        if(
            !trace.experienceUsage
                .skillIds
                .includes(id)
        ){


            trace.experienceUsage
                .skillIds
                .push(id);



            trace.experienceUsage
                .skills
                .push(skill);

        }

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

){


    const meta =

        result?.executionMeta;



    if(!meta)
        return;



    if(
        !trace.statistics ||
        typeof trace.statistics !== "object"
    ){

        trace.statistics = {

            attempts:0,

            retries:0,

            replans:0

        };

    }



    trace.statistics.retries =

        Number(

            meta.retryCount || 0

        );



    trace.statistics.replans =

        Number(

            meta.replanCount || 0

        );

}



/*
 * =========================================================
 * FAILURE
 * =========================================================
 */


function collectFailure(

    trace,

    result

){


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
 * SINGLE RESULT
 * =========================================================
 */


export function updateTraceFromResult(

    trace,

    result

){


    if(
        !trace ||
        !result
    ){

        return trace;

    }



    trace.result = {

        ...result

    };



    trace.validation =

        result.validation ||

        null;



    trace.terminal =

        result.terminal ||

        null;



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

){


    if(
        !trace ||
        !summary
    ){

        return trace;

    }



    trace.result = {

        ...summary

    };



    trace.validation =

        summary.validation ||

        null;



    trace.terminal =

        summary.terminal ||

        null;



    const results =

        safeArray(

            summary.results

        );



    addTraceEvent(

        trace,

        "SUMMARY",

        {

            status:

                summary.status ||
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
     * =====================================================
     * CHILD RESULTS
     * =====================================================
     */


    for(
        const result of results
    ){


        if(
            !result ||
            typeof result !== "object"
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

    }



    return trace;

}
