/*
 * =========================================================
 * JESSICA STARTUP AUTONOMOUS LEARNING E2E TEST v1
 * =========================================================
 *
 * Проверяет новую автономную цепочку:
 *
 * Synthetic Execution Trace
 *          ↓
 * Real Learning Trigger
 *          ↓
 * Learning Queue
 *          ↓
 * Learning Worker
 *          ↓
 * Candidate / Proposal
 *          ↓
 * Atomic Proposal Claim
 *          ↓
 * PROCESSING
 *          ↓
 * Reviewer
 *          ↓
 * Quality Gate
 *          ↓
 * Autonomy
 *          ↓
 * Experience Persistence
 *          ↓
 * published Experience
 *
 *
 * Запускается только:
 *
 * RUN_AUTONOMOUS_LEARNING_E2E_TEST_ON_START=true
 *
 * =========================================================
 */


import {
    runLearningDaemonCycle
} from "../core/learning/learningDaemon.js";


import {
    createAutonomousLearningTestTrace
} from "./autonomousLearningTest/autonomousTestTrace.js";


import {
    createAutonomousLearningTestEvent
} from "./autonomousLearningTest/autonomousTestEvent.js";


import {
    ensureAutonomousTestCanRun,
    queueAutonomousTestEvent,
    inspectAutonomousTestState,
    cleanupAutonomousTestSkill
} from "./autonomousLearningTest/autonomousTestStorage.js";


import {
    assertAutonomousLearningResult
} from "./autonomousLearningTest/autonomousTestAssertions.js";


import {
    AUTONOMOUS_TEST_FLAG,
    AUTONOMOUS_TEST_SKILL_ID
} from "./autonomousLearningTest/autonomousTestConstants.js";





function isTestEnabled()
{

    return (

        String(

            process.env[
                AUTONOMOUS_TEST_FLAG
            ]

            ||

            ""

        )
        .trim()
        .toLowerCase()

        ===

        "true"

    );

}





export async function runStartupAutonomousLearningTest()
{

    if(
        !isTestEnabled()
    ){

        return;

    }


    console.log(
        "========================================"
    );

    console.log(
        "Jessica Autonomous Learning E2E: START"
    );

    console.log(
        "Diagnostic Skill:",
        AUTONOMOUS_TEST_SKILL_ID
    );

    console.log(
        "========================================"
    );


    let queueItem = null;


    try {


        /*
         * =================================================
         * 1. REPEAT PROTECTION
         * =================================================
         */


        const eligibility =

            await ensureAutonomousTestCanRun();


        if(
            !eligibility.canRun
        ){

            console.log(
                "Jessica Autonomous Learning E2E: SKIPPED"
            );

            console.log(
                "Diagnostic Experience History уже существует:",
                eligibility.history.length
            );

            console.log(
                "Используй новый diagnostic skill ID для повторного полного теста."
            );


            return;

        }


        /*
         * =================================================
         * 2. SYNTHETIC EXECUTION
         * =================================================
         */


        const trace =

            createAutonomousLearningTestTrace();


        /*
         * =================================================
         * 3. REAL ANALYZER / TRIGGER
         * =================================================
         */


        const eventResult =

            createAutonomousLearningTestEvent(
                trace
            );


        if(
            !eventResult.success
        ){

            throw new Error(

                eventResult.reason

                ||

                "Learning Trigger diagnostic failed"

            );

        }


        console.log(

            "Jessica Autonomous E2E Trigger:",

            JSON.stringify(
                {
                    action:
                        eventResult.event?.action,

                    confidence:
                        eventResult.event?.confidence,

                    reason:
                        eventResult.event?.reason
                },
                null,
                2
            )

        );


        /*
         * =================================================
         * 4. QUEUE
         * =================================================
         */


        const queued =

            await queueAutonomousTestEvent(
                eventResult.event
            );


        if(
            !queued.success
        ){

            throw new Error(

                queued.error

                ||

                "Diagnostic Queue failed"

            );

        }


        queueItem =
            queued.queueItem;


        console.log(
            "Jessica Autonomous E2E Queue Item:",
            queueItem.id
        );


        /*
         * =================================================
         * 5. RUN REAL DAEMON CYCLE
         * =================================================
         */


        const daemonResult =

            await runLearningDaemonCycle();


        console.log(

            "Jessica Autonomous E2E Daemon:",

            JSON.stringify(
                {
                    success:
                        daemonResult?.success,

                    created:
                        daemonResult?.created,

                    pending:
                        daemonResult?.pending,

                    proposals:
                        daemonResult?.proposals,

                    learned:
                        daemonResult?.learned,

                    candidates:
                        daemonResult?.candidates,

                    rejected:
                        daemonResult?.rejected,

                    failed:
                        daemonResult?.failed
                },
                null,
                2
            )

        );


        /*
         * =================================================
         * 6. STORAGE INSPECTION
         * =================================================
         */


        const state =

            await inspectAutonomousTestState(
                queueItem.id
            );


        /*
         * =================================================
         * 7. ASSERTIONS
         * =================================================
         */


        const assertions =

            assertAutonomousLearningResult({

                daemonResult,

                queueItem,

                state

            });


        console.log(

            "Jessica Autonomous Learning E2E assertions:"

        );


        console.log(

            JSON.stringify(
                assertions,
                null,
                2
            )

        );


        if(
            assertions.success !== true
        ){

            throw new Error(

                `Autonomous Learning E2E failed: ${assertions.failed} assertions failed`

            );

        }


        console.log(
            "Jessica Autonomous Learning E2E: SUCCESS"
        );


    }catch(error){


        console.error(

            "Jessica Autonomous Learning E2E failed:",

            error

        );


    }finally{


        /*
         * =================================================
         * 8. CLEANUP ACTIVE MEMORY
         * =================================================
         *
         * History намеренно сохраняется
         * как audit proof.
         *
         * Current Experience выключаем,
         * чтобы диагностический Skill
         * не участвовал в реальных запросах.
         *
         * =================================================
         */


        try {


            const cleanup =

                await cleanupAutonomousTestSkill();


            console.log(

                "Jessica Autonomous Learning E2E cleanup:",

                JSON.stringify(
                    cleanup,
                    null,
                    2
                )

            );


        }catch(error){


            console.error(

                "Jessica Autonomous Learning E2E cleanup failed:",

                error

            );

        }


        console.log(
            "========================================"
        );

        console.log(
            "Jessica Autonomous Learning E2E: END"
        );

        console.log(
            "========================================"
        );

    }

}
