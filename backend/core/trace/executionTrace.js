/*
 * =========================================================
 * JESSICA EXECUTION TRACE v7
 * =========================================================
 *
 * История одного Execution Run.
 *
 *
 * Отвечает:
 *
 * - события Execution;
 * - попытки;
 * - шаги;
 * - ошибки;
 * - Result;
 * - Experience usage;
 * - Learning payload.
 *
 *
 * НЕ:
 *
 * - выполняет задачи;
 * - принимает решения;
 * - меняет Skills.
 *
 * =========================================================
 */


import {
    randomUUID
} from "node:crypto";









function safeArray(value){

    return Array.isArray(value)

        ?

        value

        :

        [];

}









/*
 * =========================================================
 * CREATE TRACE
 * =========================================================
 */


export function createExecutionTrace(

    task

){


    return {


        id:

            randomUUID(),



        task:

            String(

                task || ""

            )
            .trim(),



        startedAt:

            new Date()
            .toISOString(),



        finishedAt:

            null,



        status:

            "RUNNING",







        events:[],


        attempts:[],


        steps:[],


        failures:[],


        replans:[],







        result:null,



        validation:null,



        terminal:null,









        contextSnapshot:

        {

            executionId:null,


            initialPlan:null,


            currentPlan:null


        },









        experienceUsage:

        {

            used:false,


            source:null,


            confidence:0,


            skills:[],


            skillIds:[]

        },









        statistics:

        {

            attempts:0,


            retries:0,


            replans:0


        },









        learningReady:false


    };


}









/*
 * =========================================================
 * EVENT
 * =========================================================
 */


export function addTraceEvent(

    trace,

    type,

    payload={}

){


    if(!trace)
        return trace;



    if(!Array.isArray(trace.events))
        trace.events=[];



    trace.events.push({

        id:

            randomUUID(),



        type,



        payload,



        timestamp:

            new Date()
            .toISOString()

    });



    return trace;

}









/*
 * =========================================================
 * ATTEMPT
 * =========================================================
 */


export function addTraceAttempt(

    trace,

    attempt,

    data={}

){


    if(!trace)
        return trace;



    trace.statistics.attempts =

        Number(attempt);



    trace.attempts.push({

        attempt,


        ...data,


        timestamp:

            new Date()
            .toISOString()


    });



    return trace;

}









/*
 * =========================================================
 * RESULT
 * =========================================================
 */


export function updateTraceFromResult(

    trace,

    result

){


    if(

        !trace ||

        !result

    )
        return trace;









    trace.result =

    {

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









    if(result.failure){


        trace.failures.push({

            ...result.failure,


            timestamp:

                new Date()
                .toISOString()

        });


    }









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

            getSkillId(skill);



        if(!id)
            continue;





        if(

            !trace.experienceUsage.skillIds.includes(id)

        ){


            trace.experienceUsage.skillIds.push(id);


            trace.experienceUsage.skills.push(skill);


        }


    }


}









function getSkillId(skill){

    if(

        typeof skill === "string"

    )

        return skill;



    return (

        skill?.id ||

        skill?.name ||

        ""

    );

}









/*
 * =========================================================
 * META
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
 * REPLAN
 * =========================================================
 */


export function addTraceReplan(

    trace,

    data={}

){


    if(!trace)
        return trace;



    const item = {


        previousPlan:

            data.previousPlan || null,



        newPlan:

            data.newPlan || null,



        failure:

            data.failure || null,



        timestamp:

            new Date()
            .toISOString()


    };



    trace.replans.push(item);



    addTraceEvent(

        trace,

        "REPLAN",

        item

    );



    return trace;

}









/*
 * =========================================================
 * FINISH
 * =========================================================
 */


export function finishExecutionTrace(

    trace

){


    if(!trace)
        return trace;



    trace.finishedAt =

        new Date()
        .toISOString();









    switch(

        trace.result?.status

    ){


        case "COMPLETED":

            trace.status="COMPLETED";

            break;



        case "NO_VERIFIED_RESULT":

            trace.status="NO_VERIFIED_RESULT";

            break;



        case "NEEDS_CLARIFICATION":

            trace.status="NEEDS_CLARIFICATION";

            break;



        default:

            trace.status="FAILED";


    }









    trace.learningReady=true;



    return trace;

}









/*
 * =========================================================
 * LEARNING PAYLOAD
 * =========================================================
 */


export function buildLearningPayload(

    trace

){


    if(!trace)
        return null;



    return {


        executionId:

            trace.id,


        task:

            trace.task,


        status:

            trace.status,



        statistics:

            trace.statistics,



        experience:

            trace.experienceUsage,



        failures:

            trace.failures,



        replans:

            trace.replans,



        terminal:

            trace.terminal,



        steps:

            trace.steps,



        events:

            trace.events,



        result:

            trace.result


    };

}
