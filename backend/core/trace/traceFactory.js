/*
 * =========================================================
 * JESSICA TRACE FACTORY v1
 * =========================================================
 *
 * Создание Execution Trace.
 *
 *
 * Ответственность:
 *
 * - создать новый Trace;
 * - определить начальную структуру;
 * - определить начальные значения.
 *
 *
 * НЕ:
 *
 * - записывает события;
 * - записывает Result;
 * - завершает Trace;
 * - создаёт Learning Payload.
 *
 * =========================================================
 */


import {
    randomUUID
} from "node:crypto";



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



        /*
         * =================================================
         * HISTORY
         * =================================================
         */


        events:[],


        attempts:[],


        steps:[],


        failures:[],


        replans:[],



        /*
         * =================================================
         * RESULT
         * =================================================
         */


        result:null,


        validation:null,


        terminal:null,



        /*
         * =================================================
         * CONTEXT
         * =================================================
         */


        contextSnapshot:

        {

            executionId:null,


            initialPlan:null,


            currentPlan:null

        },



        /*
         * =================================================
         * EXPERIENCE
         * =================================================
         */


        experienceUsage:

        {

            used:false,


            source:null,


            confidence:0,


            skills:[],


            skillIds:[]

        },



        /*
         * =================================================
         * STATISTICS
         * =================================================
         */


        statistics:

        {

            attempts:0,


            retries:0,


            replans:0

        },



        /*
         * =================================================
         * LEARNING
         * =================================================
         */


        learningReady:false


    };


}
