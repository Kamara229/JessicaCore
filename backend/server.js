/*
 * =========================================================
 * JESSICA CORE BACKEND
 * =========================================================
 *
 * Главная точка запуска Jessica Backend.
 *
 *
 * Ответственность:
 *
 * - Environment;
 * - Express;
 * - Tools initialization;
 * - HTTP routes;
 * - HTTP server;
 * - Learning Daemon;
 * - startup diagnostics;
 * - graceful shutdown.
 *
 *
 * Вся бизнес-логика находится
 * в специализированных модулях.
 *
 * =========================================================
 */


/*
 * =========================================================
 * ENVIRONMENT
 * =========================================================
 */


import "dotenv/config";


/*
 * =========================================================
 * EXPRESS
 * =========================================================
 */


import express from "express";


/*
 * =========================================================
 * TOOLS
 * =========================================================
 */


import "./tools/initTools.js";


import {
    listTools
} from "./tools/toolRegistry.js";


/*
 * =========================================================
 * ROUTES
 * =========================================================
 */


import {
    registerRoutes
} from "./http/routes/registerRoutes.js";


/*
 * =========================================================
 * LEARNING DAEMON
 * =========================================================
 */


import {
    startLearningDaemon,
    stopLearningDaemon
} from "./core/learning/learningDaemon.js";


/*
 * =========================================================
 * STARTUP DIAGNOSTICS
 * =========================================================
 */


import {
    runStartupLearningTest
} from "./scripts/startupLearningTest.js";


import {
    runStartupLearningApprovalTest
} from "./scripts/startupLearningApprovalTest.js";





/*
 * =========================================================
 * APPLICATION
 * =========================================================
 */


const app =
    express();





/*
 * =========================================================
 * JSON
 * =========================================================
 */


app.use(

    express.json({

        limit:
            "1mb"

    })

);





/*
 * =========================================================
 * ROUTES
 * =========================================================
 */


registerRoutes(
    app
);





/*
 * =========================================================
 * SERVER CONFIG
 * =========================================================
 */


const port =

    Number(
        process.env.PORT
    )

    ||

    3000;


/*
 * Render Web Service должен видеть
 * внешний TCP listener.
 */


const host =

    process.env.HOST

    ||

    "0.0.0.0";





/*
 * =========================================================
 * LEARNING DAEMON CONFIG
 * =========================================================
 */


function resolveLearningDaemonInterval()
{

    const configured =

        Number(
            process.env.LEARNING_DAEMON_INTERVAL_MS
        );


    if(
        Number.isFinite(
            configured
        )
        &&
        configured >= 60_000
    ){

        return configured;

    }


    /*
     * Default:
     *
     * 15 minutes
     */


    return 15 * 60 * 1000;

}





/*
 * =========================================================
 * STARTUP DIAGNOSTICS
 * =========================================================
 */


function runStartupDiagnostics()
{

    /*
     * Safe Learning diagnostic.
     *
     * RUN_LEARNING_TEST_ON_START=true
     */


    runStartupLearningTest()

        .catch(
            error => {


                console.error(

                    "Jessica startup Learning diagnostic error:",

                    error

                );

            }
        );


    /*
     * Full Approval diagnostic.
     *
     * RUN_LEARNING_APPROVAL_TEST_ON_START=true
     *
     * Может реально записывать Experience
     * в Supabase, поэтому обычно false.
     */


    runStartupLearningApprovalTest()

        .catch(
            error => {


                console.error(

                    "Jessica startup Learning Approval diagnostic error:",

                    error

                );

            }
        );

}





/*
 * =========================================================
 * START HTTP SERVER
 * =========================================================
 */


const server =

    app.listen(

        port,

        host,

        () => {


            /*
             * =============================================
             * HTTP
             * =============================================
             */


            console.log(

                `Jessica Core started on ${host}:${port}`

            );


            /*
             * =============================================
             * TOOLS
             * =============================================
             */


            const tools =

                listTools()

                    .map(
                        tool =>
                            tool.name
                    );


            console.log(

                `Jessica tools (${tools.length}):`,

                tools.join(
                    ", "
                )

            );


            /*
             * =============================================
             * LEARNING DAEMON
             * =============================================
             */


            const learningInterval =

                resolveLearningDaemonInterval();


            const daemon =

                startLearningDaemon(
                    learningInterval
                );


            if(
                daemon?.started === true
            ){

                console.log(

                    `Jessica Learning Daemon interval: ${learningInterval} ms`

                );

            }else{


                console.warn(

                    "Jessica Learning Daemon was not started:",

                    daemon?.reason

                    ||

                    "unknown reason"

                );

            }


            /*
             * =============================================
             * DIAGNOSTICS
             * =============================================
             */


            runStartupDiagnostics();

        }

    );





/*
 * =========================================================
 * SERVER ERROR
 * =========================================================
 */


server.on(

    "error",

    error => {


        console.error(

            "Jessica HTTP Server error:",

            error

        );

    }

);





/*
 * =========================================================
 * GRACEFUL SHUTDOWN
 * =========================================================
 */


let shuttingDown =
    false;


function shutdown(
    signal
) {

    if(
        shuttingDown
    ){

        return;

    }


    shuttingDown =
        true;


    console.log(

        `Jessica shutdown requested: ${signal}`

    );


    /*
     * Stop background Learning timer.
     */


    try {


        stopLearningDaemon();


    }catch(error){


        console.error(

            "Jessica Learning Daemon stop error:",

            error

        );

    }


    /*
     * Stop accepting new HTTP requests.
     */


    server.close(

        error => {


            if(
                error
            ){

                console.error(

                    "Jessica HTTP Server shutdown error:",

                    error

                );


                process.exitCode =
                    1;

            }


            console.log(

                "Jessica Core stopped"

            );

        }

    );

}


/*
 * Render normally sends SIGTERM
 * before stopping/redeploying instance.
 */


process.on(

    "SIGTERM",

    () =>
        shutdown(
            "SIGTERM"
        )

);


process.on(

    "SIGINT",

    () =>
        shutdown(
            "SIGINT"
        )

);
