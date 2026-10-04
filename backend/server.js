/*
 * =========================================================
 * JESSICA CORE BACKEND v2
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
 *
 * Environment должен быть загружен
 * раньше остальных модулей Jessica.
 *
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
 * TOOLS INITIALIZATION
 * =========================================================
 *
 * Импорт регистрирует инструменты
 * в Tool Registry.
 *
 * =========================================================
 */


import "./tools/initTools.js";


import {
    listTools
} from "./tools/toolRegistry.js";


/*
 * =========================================================
 * HTTP ROUTES
 * =========================================================
 */


import {
    registerRoutes
} from "./http/routes/registerRoutes.js";


/*
 * =========================================================
 * LEARNING DAEMON
 * =========================================================
 *
 * Автономный фоновый Learning Pipeline.
 *
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


/*
 * Correction Learning diagnostic.
 *
 * Не сохраняет Experience.
 */


import {
    runStartupLearningTest
} from "./scripts/startupLearningTest.js";


/*
 * Старый correction-based
 * Approval diagnostic.
 *
 * Может сохранять Experience.
 */


import {
    runStartupLearningApprovalTest
} from "./scripts/startupLearningApprovalTest.js";


/*
 * Новый Autonomous Learning E2E.
 *
 * Проверяет цепочку:
 *
 * Execution Trace
 *      ↓
 * Learning Trigger
 *      ↓
 * Queue
 *      ↓
 * Worker
 *      ↓
 * Proposal
 *      ↓
 * Atomic Claim
 *      ↓
 * Approval
 *      ↓
 * Experience
 *
 * Запускается только отдельным
 * environment flag.
 */


import {
    runStartupAutonomousLearningTest
} from "./scripts/startupAutonomousLearningTest.js";





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
 * Render Web Service должен
 * слушать внешний интерфейс.
 */


const host =

    String(

        process.env.HOST

        ||

        "0.0.0.0"

    )
    .trim()

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


    /*
     * Минимум 60 секунд.
     *
     * Это защита от случайной
     * слишком частой конфигурации.
     */


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
     * 15 minutes.
     */


    return 15 * 60 * 1000;

}





/*
 * =========================================================
 * STARTUP DIAGNOSTICS
 * =========================================================
 *
 * Каждый диагностический тест
 * самостоятельно проверяет свой флаг.
 *
 *
 * Поэтому вызовы безопасно присутствуют
 * постоянно.
 *
 * =========================================================
 */


function runStartupDiagnostics()
{

    /*
     * =====================================================
     * 1. CORRECTION LEARNING TEST
     * =====================================================
     *
     * ENV:
     *
     * RUN_LEARNING_TEST_ON_START=true
     *
     *
     * Experience НЕ сохраняется.
     *
     * =====================================================
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
     * =====================================================
     * 2. CORRECTION APPROVAL TEST
     * =====================================================
     *
     * ENV:
     *
     * RUN_LEARNING_APPROVAL_TEST_ON_START=true
     *
     *
     * ВАЖНО:
     *
     * этот тест может реально
     * сохранить Experience.
     *
     * Обычно должен быть false.
     *
     * =====================================================
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


    /*
     * =====================================================
     * 3. AUTONOMOUS LEARNING E2E
     * =====================================================
     *
     * ENV:
     *
     * RUN_AUTONOMOUS_LEARNING_E2E_TEST_ON_START=true
     *
     *
     * Проверяет новую автономную цепочку:
     *
     * Synthetic Execution Trace
     *          ↓
     * Learning Trigger
     *          ↓
     * Learning Queue
     *          ↓
     * Worker
     *          ↓
     * Candidate / Proposal
     *          ↓
     * Atomic Claim
     *          ↓
     * PROCESSING
     *          ↓
     * Reviewer / Quality / Autonomy
     *          ↓
     * Experience Persistence
     *          ↓
     * published Experience
     *
     *
     * По умолчанию должен быть false.
     *
     * =====================================================
     */


    runStartupAutonomousLearningTest()

        .catch(
            error => {


                console.error(

                    "Jessica Autonomous Learning E2E startup error:",

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
             * HTTP READY
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


            const daemonResult =

                startLearningDaemon(
                    learningInterval
                );


            if(
                daemonResult?.started === true
            ){

                console.log(

                    `Jessica Learning Daemon interval: ${learningInterval} ms`

                );

            }else{


                console.warn(

                    "Jessica Learning Daemon was not started:",

                    daemonResult?.reason

                    ||

                    "unknown reason"

                );

            }


            /*
             * =============================================
             * STARTUP DIAGNOSTICS
             * =============================================
             *
             * Запускаем только после:
             *
             * - открытия HTTP port;
             * - инициализации Tools;
             * - запуска Learning Daemon.
             *
             * =============================================
             */


            runStartupDiagnostics();

        }

    );





/*
 * =========================================================
 * HTTP SERVER ERROR
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

    /*
     * Повторный сигнал
     * не должен запускать shutdown дважды.
     */


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
     * =====================================================
     * STOP LEARNING DAEMON
     * =====================================================
     */


    try {


        const daemonResult =

            stopLearningDaemon();


        if(
            daemonResult?.stopped === true
        ){

            console.log(

                "Jessica Learning Daemon stopped"

            );

        }


    }catch(error){


        console.error(

            "Jessica Learning Daemon stop error:",

            error

        );

    }


    /*
     * =====================================================
     * STOP HTTP SERVER
     * =====================================================
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

                return;

            }


            console.log(

                "Jessica Core stopped"

            );

        }

    );

}





/*
 * =========================================================
 * PROCESS SIGNALS
 * =========================================================
 *
 * Render при redeploy / shutdown
 * обычно отправляет SIGTERM.
 *
 * SIGINT нужен для локального запуска.
 *
 * =========================================================
 */


process.on(

    "SIGTERM",

    () => {

        shutdown(
            "SIGTERM"
        );

    }

);


process.on(

    "SIGINT",

    () => {

        shutdown(
            "SIGINT"
        );

    }

);
